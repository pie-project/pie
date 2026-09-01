//! What the shell refuses, and whose fault it is.
//!
//! THREE ERROR VOCABULARIES MEET HERE AND NONE OF THEM ABSORBS ANOTHER.
//! [`KernelError`] is about the backend and never about the plan;
//! [`model_exec::Error`] is about a fire the artifact cannot describe; and this
//! type is about everything that happens *around* a fire — binding a device,
//! landing a checkpoint, carving bytes. Folding any of the three into another
//! would send an operator hunting for a missing kernel when what actually
//! failed was a `cudaMalloc`, which is the failure mode the split exists to
//! prevent.
//!
//! A FOURTH NOW ARRIVES FROM BELOW and is translated rather than carried:
//! `kernels_cuda::Error`, which the kernel library kept when the crate the two
//! shared was taken apart. [`kernel`] at the foot of this file is that seam,
//! and its doc says why it is a function where a `From` impl would read
//! better.
//!
//! An integrity break — a weight row the shell never bound, a cache id in a
//! tensor seat — is NOT here: [`Run`](crate::run::Run) panics with a sentence
//! for those, because they are bugs in this crate rather than conditions a
//! caller can act on. What reaches this enum is always something the caller
//! did, or something the machine did.

use std::fmt;

use model_exec::KernelError;

/// The shell's result.
pub type Result<T> = std::result::Result<T, Fault>;

/// One refusal, naming what it was and the numbers behind it.
#[derive(Debug)]
pub enum Fault {
    /// This build selected no CUDA runtime.
    ///
    /// The same answer `kernels-cuda` gives a fire without one, restated at
    /// the layer that would have allocated the bytes — a build with no
    /// `cuda-12`/`cuda-13` compiles the whole shell and reaches a device at
    /// exactly one place, which is here.
    Runtimeless,

    /// A CUDA runtime call answered with an error code.
    Device {
        /// The entry point, as the runtime spells it.
        call: &'static str,
        /// Its status code.
        code: i32,
    },

    /// The compiler refused to bake this plan against these budgets.
    Bake(model_compiler::Error),

    /// The loader refused to land this checkpoint.
    Load(checkpoint::error::Error),

    /// The fire substrate refused this batch, or the backend refused a
    /// dispatch inside it.
    ///
    /// **IT USED TO CARRY A THIRD THING AND SHOULD NOT HAVE.** The substrate
    /// was one crate with one `Error` enum, and its `Program { message }`
    /// variant — a launch program the ETA interpreter could not read — arrived
    /// here as a `Fire` like any other. `fault()` sorts `Fire` to
    /// `Error::Invalid` and the guest-program refusals to `Error::Program`, so
    /// every interpreter refusal crossed the contract wearing the wrong noun,
    /// and a holder of this variant had to match an inner enum to find out
    /// which. The substrate is two crates now and so is this: what is left
    /// here is a batch the artifact cannot describe, and nothing else.
    Fire(model_exec::Error),

    /// A region whose classes this fire's order does not make consecutive, and
    /// which the artifact owes no answer for.
    ///
    /// **P4's PROMISE, FOUND BROKEN — AND NOT MERELY UNKEPT.** The layout pass
    /// solves one global C1P instance so that every windowed consumer's class
    /// set is an interval of the class order (design §3), which is what lets a
    /// windowed kernel take a pointer and an extent. When it cannot seat a
    /// consumer it says so, in
    /// [`FallbackTable`](model_compiler::FallbackTable) — and THAT is a slow
    /// path, not this: `model_exec::fire::walk` dispatches such a region once per
    /// interval and `crate::window::Windows` cuts it a window per launch. The
    /// catalog does bake rows (the four qwen texts owe 12–84 apiece, all of
    /// them the `captures_scores` window), so an empty table is not the
    /// premise here and never was.
    ///
    /// What is left for this variant is the case where the two halves
    /// disagree: a mask P4 PROMISED consecutive that this fire found in
    /// pieces, or one whose pieces outnumber the `Fallback::Split { r }` P4
    /// counted on the order it shipped. A fire's class order is that order
    /// with the absent classes dropped and dropping a class can only close a
    /// gap, so neither can happen to a `CompiledModel` and a `WindowTable` built from
    /// each other. Both are refused by name rather than run over the classes
    /// in between.
    Fragmented {
        /// Which region of `CompiledModel::template`.
        region: u32,
        /// How many runs its mask covers in this fire.
        runs: usize,
        /// How many P4 wrote down, or `None` when it wrote nothing at all —
        /// which is the promise being broken rather than exceeded.
        promised: Option<u32>,
    },

    /// A tensor the checkpoint published that this plan does not name, or a
    /// plan param the checkpoint never published.
    Param {
        /// The name, as the plan and the load contract both spell it.
        name: String,
        /// Which side was missing it.
        why: &'static str,
    },

    /// **A weight-residency budget this plan cannot be served under** (alto
    /// design §7).
    ///
    /// `Impossible`, never `Exhausted`: nothing the deployment frees changes
    /// the answer, because the refusal is about a TIER this build does not
    /// have for those planes rather than about a pool that is full. The one
    /// tier that exists is the routed-expert one — a device slab smaller than
    /// a bank, over pinned host bytes — so a budget under the DENSE planes,
    /// or a budget under a plan with no routed bank at all, lands here with
    /// both numbers in the sentence.
    Residency(String),

    /// A count past a ceiling the shell reserved bytes for.
    /// A media submission whose patch payload does not match the geometry
    /// beside it.
    ///
    /// **REFUSAL (i) OF MULTIMODAL M-1e, IN BYTES.** `Media::rows` says how
    /// many patch rows each image contributes and `Media::patches` carries
    /// them; the plan states one row's width. Three numbers that must agree,
    /// and a submission where they do not has a caller who packed one image's
    /// grid and described another's — which does not fault on the device, it
    /// reads whatever the next image's rows are.
    PatchPayload {
        /// Which lane of the submission.
        lane: u32,
        /// What its geometry adds up to, in bytes.
        need: u64,
        /// What its payload carries.
        have: u64,
    },

    Ceiling {
        /// What overflowed.
        what: &'static str,
        /// What was asked.
        need: u64,
        /// What was reserved.
        have: u64,
    },

    /// A lane's mask does not REACH the lane's readable extent.
    ///
    /// **THE SHORT DIRECTION IS THE ONLY DANGEROUS ONE.** A mask states
    /// `total` positions and the lane's readable extent after this fire's
    /// append is `held + rows`; a mask that covers fewer positions than that
    /// expands into a rectangle whose tail bits are zero, and zero is
    /// MASKED-OUT — a silently truncated attention rather than a fault. So a
    /// short mask is refused and is not padded.
    ///
    /// **A LONGER MASK IS NOT THIS FAULT** and was, until the per-row wave,
    /// wrongly folded into it. A guest builds its mask over the pages it
    /// RESERVED — 48 keys for a 3-page pool holding 23 tokens — because that
    /// is the width it knows before the sequence has a length, and every
    /// masked inferlet in `tests/inferlets` states one that way. The surplus
    /// is unambiguous: a position past `held + rows` is one this fire has not
    /// written, so the causal bound drops it for every query row whatever its
    /// bit says. It is clipped rather than refused, which is also how the C++
    /// engine this replaced read one (`brle::decode` walked `0..kv_len` and
    /// read past the row's words as false).
    Mask {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// How many positions the mask says it covers.
        stated: u64,
        /// How many the lane will hold once this fire's tokens are written.
        extent: u64,
    },

    /// A lane's per-row mask does not have one mask per query row.
    ///
    /// **THE ROW AXIS IS THE POINT OF THE FORM, SO A SHORT ONE IS NOT A
    /// MASK.** `Masking::Rows` states one restriction per query row and this
    /// shell walks them in step with the rows — row `q`'s runs under row
    /// `q`'s causal bound. A list that is not the lane's length leaves some
    /// row with no restriction of its own, and the only ways to proceed are
    /// to invent one (which is the silent row-ZERO substitution the per-row
    /// form exists to end) or to leave the row's bits clear (which is an
    /// empty softmax, MASKED-OUT everywhere, and a blanked logit rather than
    /// a fault). Neither is served.
    ///
    /// `Lane::validate` refuses the same shape at the contract door; this is
    /// the same rule where the expansion is, because a `Seated` can be built
    /// without a `Lane` ever crossing the api.
    MaskRows {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// How many per-row masks the lane stated.
        stated: u64,
        /// How many token rows this fire feeds the lane.
        rows: u32,
    },

    /// A masked lane in a fire whose loaded artifact bakes no masked class.
    ///
    /// **A DECLARED AXIS IS THE MODEL'S, NOT THE SUBMISSION'S** (design §8).
    /// A mask is a runtime input, but the node that READS one is a supergraph
    /// arm the model text either declares or does not: only a plan carrying
    /// `attention.masked` has anywhere for the bits to go. Accepting a mask
    /// against a plan without one would stage bytes no launch reads and
    /// answer with the unmasked continuation, which is a wrong answer that
    /// looks like a right one.
    Maskless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose mask do not agree.
    ///
    /// **THE AXIS IS PER LANE, AND BOTH HALVES HAVE TO SAY SO** (design §0's
    /// vocabulary note). `masked` is a bit in the word the runtime stamps from
    /// the model's own `Classify::of`, and the mask is the bits beside it;
    /// the word chooses the CLASS and therefore the window, and the mask is
    /// what the window's launch reads. A lane whose class runs
    /// `attention.masked` and that brought no mask would send the arm at a
    /// slab this fire never staged; a lane that brought a mask and landed in
    /// a class that runs the plain arm would have its mask staged and never
    /// read, and answer with the unmasked continuation.
    ///
    /// The second direction is how a FIRE-WIDE mask presents here. A pass
    /// whose `AttnMask` port is device-resident has one dense mask for the
    /// whole fire and no per-lane state, so the runtime stamps `masked` on
    /// every lane and puts a mask on none — exactly the collapse §0 warns
    /// about, arriving as a named refusal rather than as a rectangle of
    /// zeros.
    MaskWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the masked arm.
        runs_masked_arm: bool,
    },

    /// An adapted lane in a fire whose loaded artifact bakes no correction.
    ///
    /// [`Fault::Maskless`]'s twin, for the axis beside it and for the same
    /// reason: an adapter bank is a supergraph seat the model text either
    /// declares or does not (design §8), and a lane routed against a plan
    /// with no `linear.lora_correct` arm would get the base model's answer
    /// under an adapter's name.
    Adapterless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose adapter do not agree.
    ///
    /// [`Fault::MaskWord`]'s twin. The word chooses the class and the class
    /// chooses whether this lane's rows fall inside the correction's WINDOW,
    /// which is what design §8 means by "a correction op over the adapter
    /// window". A lane inside the window with no adapter id would send the
    /// arm at a routes vector this fire never staged; a lane outside it
    /// carrying one would have its id staged and never read, and answer with
    /// the base model's continuation — which is precisely the failure
    /// decision 17 makes the capacity a budget rather than an admission cap
    /// to avoid.
    AdapterWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the correction.
        runs_correction: bool,
    },

    /// An allocation the device has no room for — the ask and the free, both
    /// in bytes (palo C3b).
    ///
    /// **AN OUT-OF-MEMORY REFUSAL IS A FACT ABOUT TWO NUMBERS, AND IT SHOULD
    /// SAY BOTH.** Every allocation this shell makes is sized once from a
    /// budget and lives until the model is unloaded, so the interesting
    /// failure is always "this model does not fit this device" — and
    /// [`Fault::Device`] answers that with `cudaMalloc answered 2`, which
    /// cannot tell a shortfall of six gigabytes from one of sixty and reads
    /// like every other runtime failure. `device::alloc` asks
    /// `cudaMemGetInfo` at the moment of the refusal and states what it
    /// learned.
    ///
    /// Distinct from [`Fault::Ceiling`], which is a fire wanting more than the
    /// LOAD reserved: that is a submission past a bake and this is a bake past
    /// a card.
    OutOfMemory {
        /// Bytes asked for.
        need: u64,
        /// Bytes the device had free when the ask failed.
        have: u64,
    },

    /// A drafting lane in a fire whose loaded artifact declares no draft head.
    ///
    /// [`Fault::Maskless`]'s and [`Fault::Adapterless`]'s third: an MTP head
    /// is a supergraph arm the model text either states or does not (design
    /// §8), and a lane that asked for drafts against a plan with no
    /// `model_dsl::seam::MTP` export would be handed the trunk's continuation
    /// with a draft's name on it.
    Draftless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose draft ask do not agree.
    ///
    /// [`Fault::AdapterWord`]'s twin for the export axes, and the second wrong
    /// answer is a different one because the axis carries no payload. A lane
    /// whose word puts it inside the head's window and that asked for no
    /// draft has a whole transformer block and a vocabulary-wide GEMM run over
    /// its rows into a column nobody collects — paid for and thrown away. A
    /// lane that asked and landed outside gets no draft at all, and an absent
    /// column is indistinguishable from a column of zeros to the reader that
    /// comes for it.
    DraftWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the draft head.
        runs_draft_arm: bool,
    },

    /// A capturing lane in a fire whose loaded artifact declares no capture
    /// arm — and the refusal a score READ takes when the plan states no
    /// `attn.scores` export at all (design §9, palo C4).
    Scoreless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose capture ask do not agree.
    ///
    /// [`Fault::DraftWord`]'s twin, one axis over. A capturing word with no
    /// ask behind it runs the `attention.prefill_lse` arm and writes a mass
    /// column the readout never copies; an ask with a plain word lands the
    /// lane on the decode or prefill kernel, which produces no mass, and the
    /// caller is handed an empty capture it cannot tell from a captured
    /// nothing.
    ScoreWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the capture arm.
        runs_capture_arm: bool,
    },

    /// A registration this load's banks cannot seat.
    ///
    /// **THE BUDGET IS THE SHAPE, SO A REFUSAL CARRIES NUMBERS** (design §8,
    /// decision 17). An id past the bank's first axis, a plane whose bytes
    /// are not the slot's, a name no `ParamSource::Registered` param carries:
    /// each one is a caller and a model text that were not written from each
    /// other, and each is refused at the door rather than written past a
    /// slot's end.
    Adapter {
        /// The bank the registration named.
        bank: String,
        /// What is wrong with it.
        why: String,
    },

    /// A shared adapter the mount cannot serve (alto adapter §3.3, §5).
    ///
    /// **FILES ARE THE TRUTH, SO THE REFUSALS ARE ABOUT FILES.** An
    /// unmounted shell, a name that leaves the mount, a directory with no
    /// `adapter.toml`, a manifest naming a plane that is not there, a source
    /// whose orientation is not the bank's (§6.3's out-major statute), a
    /// length that is not `layers x rank x hidden`: each is a mount and a
    /// model text that were not written from each other, each names the
    /// adapter, and none of them falls back to anything.
    Blob {
        /// The adapter, as the bind spelled it.
        path: String,
        /// What is wrong with it.
        why: String,
    },

    /// Every adapter slot pinned by a live bind when a load wanted one.
    ///
    /// **THE ONE PRESSURE THIS AXIS ANSWERS WITH A REFUSAL** (alto adapter
    /// §3.3, §5). Residency is reclaimed LRU from slots NO bind holds; a
    /// table in which every slot is held has nothing reclaimable in it, and
    /// the alternative to refusing is moving an adapter some fire in flight
    /// routes to. `slots` bounds concurrent residency and not the catalog, so
    /// the fix is fewer live binds or a wider bank — never a retry that hopes.
    AdapterSlots {
        /// How many slots the banks seat.
        seats: u32,
    },

    /// A plan struct built over more rows than the node consuming it runs.
    ///
    /// **P4's PROMISE AT THE OTHER END** — [`Fault::Fragmented`] catches a
    /// region whose classes are not consecutive; this catches a region that
    /// is consecutive and WIDER than its reader. An attention schedule is
    /// carved at its own node's window: how many requests it batches, where
    /// each one's query rows start, how the work items are split. A consumer
    /// in a narrower window hands that schedule its own rebased boundaries,
    /// and every work item past the first request then indexes a `qo_indptr`
    /// that ends before it — reading whatever follows the vector.
    ///
    /// It happens when one plan VALUE is shared by arms in different classes:
    /// the compiler narrows a prepare node by demand to the union of the
    /// classes that read its struct, which is the right answer for a shared
    /// value and the wrong shape for two windowed readers. The fix is one
    /// plan per arm in the model text, not a fold here — so this is refused
    /// by name.
    Straddled {
        /// The plan value, as `Trace::values` numbers it.
        value: u32,
        /// The node consuming it, as `Trace::nodes` numbers it.
        node: u32,
        /// The classes its defining region runs in.
        planned: String,
        /// The classes the consuming region runs in.
        consumed: String,
    },

    /// A guest program (ETA) that does not compile on this device.
    ///
    /// **THE TAXONOMY IS THE POINT, NOT THE TEXT.**
    /// [`Deterministic`](eta_exec::Failure::Deterministic) means the source is
    /// wrong and will be wrong next time — the compile plane remembers it and
    /// answers the next registration from memory. `Retryable` means the
    /// machine could not, this time (no NVRTC, out of memory, a cubin that
    /// would not load), and remembering it would strand a program on one bad
    /// minute. Folding the two into one string is what makes an engine either
    /// retry a syntax error forever or give up on a transient.
    Compile(eta_exec::Failure),

    // A `Blocked { instance, channel }` variant stood here: an attached
    // instance whose ring was not ready, typed so `fault()` could answer
    // `Error::Exhausted` and the runtime's lane could sleep and re-submit the
    // identical frame. It was F2a's bridge to article 4 and wave E is the far
    // bank. `pipeline::fire::validate_frame` now proves ring occupancy,
    // host-writer staging and reader pressure statically at submit, so a
    // readiness miss past that door is not a scheduling answer — it is a
    // contract violation, and `serve::committed_or` says so by name. Nothing
    // in this crate answers `Exhausted` for a channel any more; `Fault::
    // OutOfMemory` remains the only exhaustion, and it is about device bytes.

    /// The guest-program plane refused this call.
    ///
    /// Its own variant rather than [`Fault::Fire`] because nothing here is a
    /// statement about the model fire: a channel index the instance does not
    /// carry, a ring longer than the lane table's pitch, a stage paired with
    /// somebody else's plan. `at` names the door and `why` the condition, in
    /// the shape the rest of this crate's refusals take.
    Program {
        /// Which entry point refused, as this crate spells it.
        at: &'static str,
        /// The condition, in a sentence.
        why: String,
    },

    /// The ETA substrate refused a launch program, in its own words.
    ///
    /// [`Fault::Program`] above is THIS crate's refusals about the guest-
    /// program plane — a door here, spelled the way the rest of this enum
    /// spells one. This is the substrate's, forwarded whole: a package whose
    /// values do not resolve, an op the interpreter does not know, a channel
    /// the plan does not declare. It is a separate variant because it is a
    /// separate author, and it carries the error rather than a string because
    /// the type is the only thing that says which author it was.
    ///
    /// Both sort to `Error::Program` at the contract, which is the point: the
    /// crossing this replaces sorted to `Error::Invalid`, by riding inside
    /// [`Fault::Fire`].
    Interpret(eta_exec::Error),

    /// A plan naming something this shell has no binding for.
    ///
    /// A refusal rather than a panic because it is a statement about the
    /// PLAN — a model this shell cannot serve yet — and the caller's recovery
    /// is to load a different one.
    Unbound {
        /// The seat, named as the IR names it.
        what: String,
    },

    /// A conditional bracket reaching a RECORDING walk that has nowhere to put
    /// it — a lowering this shell does not build, or a load that opened no
    /// conditional machinery.
    ///
    /// **THE FIRST HALF OF THIS VARIANT IS RETIRED** (graphs wave). An `IF` on
    /// a load whose context opened a body stream is now RECORDED: a real
    /// `CU_GRAPH_NODE_TYPE_CONDITIONAL` node at the capture's frontier, its
    /// body captured into the child graph the driver mints for it, and the
    /// predicate stored by a kernel — `kernels_cuda::graph::set_conditional`,
    /// reading the region's row count off the device. See
    /// [`crate::device::conditional`] for the sequence and
    /// `crate::window::Cursor::cond_begin` for where it is driven from.
    ///
    /// **AND THE LINK STAGE THIS VARIANT USED TO NAME DOES NOT EXIST.** It
    /// said `cudaGraphSetConditional` needed relocatable device code and
    /// `libcudadevrt.a`. It needs neither: the symbol is declared
    /// `extern __device__ __cudart_builtin__` and DEFINED nowhere — not in a
    /// toolkit header, not in that archive (which was extracted and searched,
    /// `.wiki/driver/new-horizon.md` §62.3) — so the driver resolves it at
    /// module load whichever frontend emitted the call. The unit compiles
    /// whole-program through the same NVRTC path as every other one.
    ///
    /// **AND THE `SWITCH` HALF IS RETIRED TOO** (B6). A group's arms are
    /// consecutive regions under one node minted with `size: arms`, the
    /// bracket lives across `arms - 1` region boundaries, and `cond_arm` is
    /// where one child graph is closed and the next begun. The predicate is
    /// one `kernels_cuda::graph::set_switch` per arm, each storing its own
    /// index only if its own window has rows — at most one does, which is the
    /// activation P3 proves before it forms a group at all.
    ///
    /// # What still answers here, and it is two things
    ///
    /// 1. **A `SWITCH` WHOSE ARM CANNOT STATE A ROW COUNT.** A region P4 split
    ///    into runs, or one on the patch axis with no boundary vector, has no
    ///    single count to read. An `IF` answers that by taking its body —
    ///    always-launch is the correctness mechanism and the guard has merely
    ///    given up an optimization. A `SWITCH` has no such direction: exactly
    ///    one body runs, so an arm guessed at is a different arm's fire
    ///    computed wrong, and the group is refused instead.
    /// 2. **A load with no body stream.** `Context::open_conditional` is
    ///    called at load only for an artifact P3 stamped a lowering on; a
    ///    cursor reaching a bracket without it is a shell being asked for
    ///    something its load did not set up.
    ///
    /// # The frozen extent, which is a bound on the SAVING and not on the node
    ///
    /// A captured launch's extent is fixed in its node parameters (build log
    /// 10), and a `record::BodyKey` names WHICH CLASSES HAVE ROWS — so a body
    /// already serves one composition and the walk skipped its empty windows
    /// at RECORD time. The predicate is therefore constant across every replay
    /// of any one body, and what the node buys there is nothing. It is not
    /// decoration: the decision is IN the graph, so an exec replayed under a
    /// composition its recording fire never saw — what a padded lattice would
    /// do, and what the retired fold's bucket axis did — skips the body
    /// correctly instead of computing over rows it does not have.
    Unlowered {
        /// Which region of the template.
        region: u32,
        /// The lowering, as the compiler spells it.
        lowering: String,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtimeless => write!(
                f,
                "this build carries no CUDA runtime: enable `cuda-12` or \
                 `cuda-13`, matching the libcudart it will load"
            ),
            Self::Device { call, code } => {
                write!(f, "{call} answered {code}")
            }
            Self::Bake(refusal) => write!(f, "this plan does not bake: {refusal:?}"),
            Self::Residency(why) => write!(f, "weight residency: {why}"),
            Self::Load(error) => write!(f, "this checkpoint does not land: {error}"),
            Self::Fire(error) => write!(f, "{error}"),
            Self::Fragmented {
                region,
                runs,
                promised,
            } => match promised {
                None => write!(
                    f,
                    "region {region} covers {runs} runs of this fire's rows and P4 \
                     wrote it no fallback row — it seriated so that this mask takes \
                     exactly one launch, and this fire's class order did not come \
                     from that seriation"
                ),
                Some(promised) => write!(
                    f,
                    "region {region} covers {runs} runs of this fire's rows where P4 \
                     counted {promised} on the order it shipped — a fire's order is \
                     that order with the absent classes dropped, and dropping a class \
                     cannot open a gap"
                ),
            },
            Self::Param { name, why } => write!(f, "`{name}` {why}"),
            Self::PatchPayload { lane, need, have } => write!(
                f,
                "lane {lane} describes {need} bytes of patch rows and submitted {have} — \
                 its geometry and its payload disagree"
            ),
            Self::Ceiling { what, need, have } => write!(
                f,
                "this fire wants {need} {what} and the shell reserved {have}"
            ),
            Self::Mask {
                lane,
                stated,
                extent,
            } => write!(
                f,
                "lane {lane}'s mask covers {stated} positions and the lane will hold \
                 {extent} once this fire's tokens are written; a mask must REACH the \
                 lane's whole readable extent, because a short one masks out the tail \
                 rather than leaving it alone (a LONGER one is fine and is clipped)"
            ),
            Self::MaskRows {
                lane,
                stated,
                rows,
            } => write!(
                f,
                "lane {lane} states {stated} per-row masks and this fire feeds it \
                 {rows} token rows; `Masking::Rows` is one restriction PER query row \
                 and a row with none of its own has no mask this shell may invent"
            ),
            Self::Maskless { lane } => write!(
                f,
                "lane {lane} carries an explicit attention mask and this load's \
                 artifact bakes no masked class: `masked` is a fact the MODEL declares \
                 (design §8) and this plan has no `attention.masked` arm for the bits \
                 to reach"
            ),
            Self::MaskWord {
                lane,
                word,
                runs_masked_arm,
            } => {
                if *runs_masked_arm {
                    write!(
                        f,
                        "lane {lane}'s word {word:#x} puts it in a class that runs \
                         `attention.masked`, and it carries no mask for that arm to \
                         read; a fire-wide mask on a per-lane axis is design \u{00A7}0's \
                         collapse, and this is where it stops"
                    )
                } else {
                    write!(
                        f,
                        "lane {lane} carries an explicit attention mask and its word \
                         {word:#x} puts it in a class that runs the causal arm, so the \
                         mask would be staged and never read"
                    )
                }
            }
            Self::Adapterless { lane } => write!(
                f,
                "lane {lane} routes to an adapter bank and this load's artifact bakes \
                 no corrected class: an adapter axis is a fact the MODEL declares \
                 (design §8) and this plan has no `linear.lora_correct` arm for the id \
                 to reach"
            ),
            Self::AdapterWord {
                lane,
                word,
                runs_correction,
            } => {
                if *runs_correction {
                    write!(
                        f,
                        "lane {lane}'s word {word:#x} puts it in a class that runs \
                         `linear.lora_correct`, and it names no adapter for that arm to \
                         route with"
                    )
                } else {
                    write!(
                        f,
                        "lane {lane} routes to an adapter and its word {word:#x} puts \
                         it in a class outside the correction's window, so the id would \
                         be staged and never read and the lane would answer with the \
                         base model"
                    )
                }
            }
            Self::OutOfMemory { need, have } => write!(
                f,
                "this load wants {need} bytes of device memory and the device has \
                 {have} free"
            ),
            Self::Draftless { lane } => write!(
                f,
                "lane {lane} asks for the model's draft head and this load's artifact declares \
                 none: an MTP axis is a fact the MODEL states (design §8) and this plan carries \
                 no `mtp` export for the readout to come from"
            ),
            Self::DraftWord {
                lane,
                word,
                runs_draft_arm,
            } => {
                if *runs_draft_arm {
                    write!(
                        f,
                        "lane {lane}'s word {word:#x} puts it in a class that runs the draft \
                         head, and it asked for no draft, so a transformer block and a \
                         vocabulary-wide readout would run over its rows into a column nobody \
                         collects"
                    )
                } else {
                    write!(
                        f,
                        "lane {lane} asks for a draft and its word {word:#x} puts it in a class \
                         outside the draft window, so no draft would be computed and the empty \
                         readout would be indistinguishable from a draft of zeros"
                    )
                }
            }
            Self::Scoreless { lane } => write!(
                f,
                "lane {lane} asks to capture its attention mass and this load's artifact \
                 declares no capture arm: a score axis is a fact the MODEL states (design §9) \
                 and this plan carries no `attn.scores` export to read"
            ),
            Self::ScoreWord {
                lane,
                word,
                runs_capture_arm,
            } => {
                if *runs_capture_arm {
                    write!(
                        f,
                        "lane {lane}'s word {word:#x} puts it in a class that runs \
                         `attention.prefill_lse`, and it asked for no capture, so the mass \
                         column would be written and never read"
                    )
                } else {
                    write!(
                        f,
                        "lane {lane} asks to capture its attention mass and its word {word:#x} \
                         puts it on the plain arm, which produces none — the empty capture \
                         cannot be told from a captured nothing"
                    )
                }
            }
            Self::Adapter { bank, why } => {
                write!(f, "the adapter bank `{bank}` {why}")
            }
            Self::Blob { path, why } => {
                write!(f, "the shared adapter `{path}` {why}")
            }
            Self::AdapterSlots { seats } => write!(
                f,
                "every one of this load's {seats} adapter slots is pinned by a live \
                 bind, and the only slot left to take would be one some fire in flight \
                 routes to — `slots` bounds concurrent residency, not the catalog, so \
                 the fix is fewer live binds or a bank that seats more"
            ),
            Self::Straddled {
                value,
                node,
                planned,
                consumed,
            } => write!(
                f,
                "value {value} is an attention schedule built over classes {planned} \
                 and node {node} consumes it over {consumed}; a schedule is carved at \
                 its own window — how many requests it batches and where each one's \
                 query rows start — so a narrower reader hands it rebased boundaries \
                 that end before its own work items do. One plan value shared by arms \
                 in different classes is narrowed to the UNION of their windows, and \
                 the fix is one plan per arm in the model text"
            ),
            Self::Compile(failure) => write!(
                f,
                "this guest program does not compile here ({}): {}",
                if failure.is_remembered() {
                    "deterministic, remembered"
                } else {
                    "retryable"
                },
                failure.reason()
            ),
            Self::Program { at, why } => write!(f, "{at}: {why}"),
            Self::Interpret(error) => write!(f, "{error}"),
            Self::Unlowered { region, lowering } => write!(
                f,
                "region {region} is baked as {lowering} and this capture has nowhere \
                 to put it: an `If` and a `Switch` both record as real conditional \
                 nodes on a load whose context opened a body stream, but a load whose \
                 artifact declared no conditional opened no stream to capture a body \
                 on, and a `Switch` arm that cannot state a row count — split into \
                 runs, or on an axis with no boundary vector — is refused rather than \
                 guessed, because exactly one arm runs and a guess is another arm's \
                 fire. Bake with `fat_region_us: INFINITY` — every region \
                 always-launch, which is the correctness mechanism"
            ),
            Self::Unbound { what } => write!(
                f,
                "this plan names {what}, which this shell does not bind"
            ),
        }
    }
}

impl std::error::Error for Fault {}

impl From<model_compiler::Error> for Fault {
    fn from(refusal: model_compiler::Error) -> Fault {
        Fault::Bake(refusal)
    }
}

impl From<checkpoint::error::Error> for Fault {
    fn from(error: checkpoint::error::Error) -> Fault {
        Fault::Load(error)
    }
}

impl From<model_exec::Error> for Fault {
    fn from(error: model_exec::Error) -> Fault {
        Fault::Fire(error)
    }
}

impl Fault {
    /// A guest-program refusal, named by its door.
    ///
    /// A constructor rather than a literal at every site: `why` is almost
    /// always a `format!`, and `Fault::Program { at, why: format!(..) }`
    /// repeated forty times is where the door name drifts from the door.
    pub(crate) fn program(at: &'static str, why: impl Into<String>) -> Fault {
        Fault::Program {
            at,
            why: why.into(),
        }
    }
}

impl From<eta_exec::Failure> for Fault {
    fn from(failure: eta_exec::Failure) -> Fault {
        Fault::Compile(failure)
    }
}

impl From<eta_exec::Error> for Fault {
    fn from(error: eta_exec::Error) -> Fault {
        Fault::Interpret(error)
    }
}

impl From<model_exec::KernelError> for Fault {
    fn from(error: model_exec::KernelError) -> Fault {
        Fault::Fire(model_exec::Error::Kernel(error))
    }
}

/// A kernel entry's refusal, reaching a shell path that answers [`Fault`].
///
/// **NOT EVERY CALL INTO `kernels-cuda` IS A DISPATCH ARM.** Weight staging,
/// the wave's control launches and the fire's own scratch work all call
/// entries there and answer `Fault`, not the contract; before the shared
/// error crate came apart this was `From<KernelError>` and `?` did the work.
/// It still can, because `Fault` is this crate's own type and the orphan rule
/// only bites when NEITHER side is — which is the whole difference between
/// this impl and [`kernel`] below.
impl From<kernels_cuda::Error> for Fault {
    fn from(error: kernels_cuda::Error) -> Self {
        Fault::from(kernel(error))
    }
}

// ---------------------------------------------------------------------------
// the seam: this backend's refusal, said in the contract's words
// ---------------------------------------------------------------------------

/// Say a [`kernels_cuda::Error`] in the dispatch contract's words.
///
/// **THIS IS A FUNCTION AND NOT A `From` IMPL, AND THAT IS NOT A STYLE
/// CHOICE.** Both types are foreign to this crate — one is the kernel
/// library's, the other the contract's — and Rust's orphan rule (E0117)
/// forbids a third crate from implementing `From` between two types it owns
/// neither of. No arrangement of these three crates gets `?` to convert here
/// without one of them naming a crate it must not: the kernel library would
/// have to depend on `model-exec`, and so on `model-compiler` and `model-ir`,
/// which is the whole edge deleting `crates/kernels` bought — or the two
/// enums would have to be one type again in a shared leaf. So the conversion
/// is called instead, once per `Dispatch*` impl, and each family's arms live
/// in an inherent method that answers [`kernels_cuda::Error`] so their own
/// `?` still converts.
///
/// **The match is total on purpose.** The two enums are variant for variant
/// identical today; they were one type until `crates/kernels` came apart, and
/// `model_exec::KernelError`'s own doc says plainly that three copies is a
/// prediction rather than a fact, with the falsifier written out. What makes
/// the prediction safe to hold is this function: the day `kernels-cuda`
/// grows the NVRTC compile-failure variant it is expected to, this stops
/// compiling, at the one line that has to decide what the new refusal means
/// to a caller who can make nothing of an NVRTC log. Nothing has to watch the
/// copy — and a copy that needs watching is the one
/// `crates/eta-exec/Cargo.toml` rules out: "a copy that is only safe because
/// something watches it is a copy that costs the watch".
pub fn kernel(error: kernels_cuda::Error) -> KernelError {
    match error {
        kernels_cuda::Error::Unsupported { op } => KernelError::Unsupported { op },
        kernels_cuda::Error::DtypeUnsupported { op, dtype } => {
            KernelError::DtypeUnsupported { op, dtype }
        }
        kernels_cuda::Error::Backend { op, detail } => KernelError::Backend { op, detail },
    }
}
