//! The shell's answer vocabulary: one enum, one `Result`, and the rule that
//! a refusal names what refused.
//!
//! The split this file keeps is the dispatch contract's, restated one layer
//! up: a [`KernelError`] is about the BACKEND — a geometry no shader is stamped
//! for, a dtype no entry carries — and a [`Fault`] is about the SHELL: a
//! device that would not allocate, a checkpoint that does not answer the
//! plan, a count past a reserved ceiling. Integrity failures of the compiler
//! or of this crate's own tables are neither; they panic with a sentence,
//! because dressing a bug as a refusal teaches a caller to retry it.
//!
//! **`Device` carries a call name and a message, where the CUDA shell's
//! twin carries a call name and an `i32`.** That is the platforms
//! disagreeing rather than the shells: a CUDA entry point returns a status
//! code from a closed ladder, and a Metal one returns `nil` beside an
//! `NSError` whose `localizedDescription` is the only description of what
//! went wrong. A shader that fails to compile says so in a paragraph with
//! line numbers in it, and throwing that away for a number would be the
//! worst trade in this file.
//!
//! **A THIRD VOCABULARY ARRIVES FROM BELOW** and is translated rather than
//! carried: `kernels_metal::Error`, which the kernel library kept when the
//! crate it shared with the contract was taken apart. [`kernel`] at the foot
//! of this file is that seam, and its doc says why it is a function where a
//! `From` impl would read better.

use std::fmt;

use model_exec::KernelError;

/// Every way this shell answers "no".
#[derive(Debug)]
pub enum Fault {
    /// This build has no Metal in it — a non-Apple target. Every device
    /// entry answers this, which is what lets the call-order code above be
    /// type-checked by a workspace sweep on Linux.
    Deviceless,

    /// A Metal call refused, and this is what it said.
    Device {
        /// The Objective-C selector, as the shell spells it.
        call: &'static str,
        /// The `NSError`'s own description, or a stated reason when the call
        /// returned `nil` without one.
        why: String,
    },

    /// A shader source or entrypoint this crate does not ship.
    Shader {
        /// The `.metal` path a `Fire` named.
        file: &'static str,
        /// The entrypoint a `Fire` named.
        entrypoint: &'static str,
        /// What went missing.
        why: String,
    },

    /// A count past a ceiling reserved at load.
    Ceiling {
        /// What was counted.
        what: &'static str,
        /// What the fire asked for.
        need: u64,
        /// What the reservation holds.
        have: u64,
    },

    /// The compiler refused to bake the plan against these budgets.
    Bake(model_compiler::Error),

    /// The loader refused to land the checkpoint against this plan.
    Load(checkpoint::error::Error),

    /// The fire substrate refused the batch, or a dispatch refused the op.
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

    /// A param the plan names that the checkpoint never published, or
    /// published at another shape.
    Param {
        /// The param's canonical name.
        name: String,
        /// Why it could not land.
        why: &'static str,
    },

    /// A region's classes are not one run AND the artifact owes it no answer —
    /// P4's promise broke, and a windowed encode would read somebody else's
    /// rows.
    ///
    /// **NOT THE SLOW PATH, WHICH IS SERVED.** P4 writes a `Fallback` row for
    /// every consumer it could not seat, and such a region is encoded once per
    /// interval of its window (`crate::window::Windows`,
    /// `model_exec::fire::fallback`). What is left here is the case where the bake
    /// and the fire disagree: a mask P4 promised consecutive that came back in
    /// pieces, or one whose pieces outnumber the `Fallback::Split { r }` P4
    /// counted on the order it shipped. Neither can happen to a `CompiledModel` and a
    /// `WindowTable` built from each other.
    Fragmented {
        /// The template region.
        region: u32,
        /// How many runs its class mask fell into in this fire.
        runs: usize,
        /// How many P4 wrote down, or `None` when it wrote nothing at all —
        /// which is the promise being broken rather than exceeded.
        promised: Option<u32>,
    },

    /// A schedule value built over one class mask and read under another.
    Straddled {
        /// The plan value holding the schedule.
        value: u32,
        /// The node that reads it.
        node: u32,
        /// The classes the schedule was planned for.
        planned: String,
        /// The classes that consume it.
        consumed: String,
    },

    /// A lane's stated mask does not REACH the extent it will read.
    ///
    /// Short only. A mask longer than the extent is the page-padded shape
    /// every masked guest states — the pool's width, not the sequence's — and
    /// is clipped, because a position past the extent is one the causal bound
    /// drops for every query row anyway. [`crate::mask`] argues it.
    Mask {
        /// The lane, in submission order.
        lane: u32,
        /// The mask's stated total.
        stated: u64,
        /// The extent the fire will read.
        extent: u64,
    },

    /// A lane carries a mask and this artifact bakes no masked class.
    ///
    /// The ARTIFACT's answer, not the plane's: `attention.masked` is a live
    /// entry here and [`crate::mask`] stages both forms, so what is missing
    /// when this is raised is a window for the rows to run in — a model text
    /// that never states the masked arm has nowhere to put a masked lane, and
    /// the unmasked continuation is the wrong answer that would look right.
    Maskless {
        /// The lane, in submission order.
        lane: u32,
    },

    /// A lane's PER-ROW mask (`Masking::Rows`) states a different number of
    /// rows than the lane feeds.
    ///
    /// **THE FORM EXPANDS; THE HEIGHT IS WHAT IS REFUSED.** A windowed
    /// prefill states one restriction per query row — row `i` keeps
    /// `[i - w, i]` — and [`crate::mask`] walks each row under its own causal
    /// bound, as the CUDA sibling does. What has no reading is a vector of
    /// some other length: `Masking::Rows` is parallel to `Lane::tokens`, so a
    /// short one would leave rows with no restriction and a long one would
    /// state restrictions for rows nobody fires. The reading it would be
    /// tempting to invent — row zero's mask on every row — is the silent
    /// substitution the whole form exists to end, so the count is named here
    /// instead. This sits BESIDE [`Fault::Maskless`] because they are
    /// different sentences: that one is about the ARTIFACT having no masked
    /// class, this one about the submission's own shape.
    MaskRows {
        /// The lane, in submission order.
        lane: u32,
        /// How many rows the masking states.
        stated: u64,
        /// How many rows the lane feeds.
        rows: u32,
    },

    /// A lane states its own token positions and states a different number
    /// of them than it feeds tokens.
    ///
    /// **THE ONLY CHECK A STATED RUN GETS, AND IT IS THE ONLY ONE THERE IS AN
    /// ANSWER TO.** Positions reach exactly one place on this plane — rope's
    /// seat, one `i32` per fire row — so a vector of the lane's height is
    /// servable whatever is in it, and the two neighbouring temptations are
    /// both wrong: padding a short one would rotate the tail at position
    /// zero, and clipping a long one would silently agree with a caller that
    /// disagrees with the composition about how many rows this lane has.
    /// Monotonicity and a page bound are NOT checked, and deliberately: the
    /// page CSR and the write descriptors are carved from `held` and the row
    /// count, never from this, so a non-monotone run is a rotation the caller
    /// meant (an mRoPE lane, a re-fed rejected draft) and not a geometry.
    Positions {
        /// The lane, in submission order.
        lane: u32,
        /// How many positions it states.
        stated: u64,
        /// How many token rows it feeds.
        rows: u64,
    },

    /// A lane's word and its mask disagree, in either direction.
    MaskWord {
        /// The lane, in submission order.
        lane: u32,
        /// The word the caller stamped.
        word: u64,
        /// Whether the word's class runs the masked arm.
        runs_masked_arm: bool,
    },

    /// An adapted lane in a fire whose loaded artifact bakes no correction.
    ///
    /// [`Fault::Maskless`]'s twin, for the axis beside it and for the same
    /// reason: an adapter bank is a seat the model text either declares or
    /// does not (design §8), and a lane routed against a plan with no
    /// `linear.lora_correct` arm would get the base model's answer under an
    /// adapter's name. Named against the BAKE rather than against the class,
    /// because when the correction is absent from the artifact no word could
    /// have put the lane inside it.
    Adapterless {
        /// The lane, in submission order.
        lane: u32,
    },

    /// A lane whose fact word and whose adapter do not agree, in either
    /// direction.
    ///
    /// [`Fault::MaskWord`]'s twin. The word chooses the class and the class
    /// chooses whether this lane's rows fall inside the correction's WINDOW,
    /// which is what design §8 means by "a correction op over the adapter
    /// window". A lane inside the window with no adapter id would send the
    /// arm at a routes vector this fire never staged; a lane outside it
    /// carrying one would have its id staged and never read, and answer with
    /// the base model's continuation under the adapter's name.
    AdapterWord {
        /// The lane, in submission order.
        lane: u32,
        /// The word the caller stamped.
        word: u64,
        /// Whether the word's class runs the correction.
        runs_correction: bool,
    },

    /// A guest program this device would not compile.
    ///
    /// Carries `eta_exec::Failure`'s own split — a `Deterministic` refusal is
    /// remembered forever (the source will not compile on this device, ever)
    /// and a `Retryable` one is not (the machine was out of something). On
    /// this plane the split comes off `MTLLibraryError`: a compile failure
    /// is the source's, and anything else is the moment's.
    Compile(eta_exec::Failure),

    /// The guest-program plane refused a call, and this is where and why.
    Program {
        /// Which entry refused, as the plane spells it.
        at: &'static str,
        /// The sentence.
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

    /// An adapter registration this load's banks cannot seat.
    ///
    /// **THE REGISTRATION'S REFUSAL, AND NOT THE AXIS'S.** A bank is a
    /// `ParamSource::Registered` weight, so [`Weights`](crate::Weights)
    /// reserves and zeroes one for any plan that declares it; this names the
    /// three ways a caller's planes fail to describe it — a bank name the
    /// plan never declared, an id past the capacity the model text stated,
    /// and a plane that is not one whole slot. The axis's own two refusals
    /// are [`Fault::Adapterless`] and [`Fault::AdapterWord`], which are about
    /// a FIRE and not about a write.
    Adapter {
        /// The bank, as the plan's param names it.
        bank: String,
        /// Why the planes do not fit it.
        why: String,
    },

    /// A quantity the ICB would have to rewrite per fire, and no affine law
    /// over the descriptor's own numbers predicts it.
    ///
    /// **THE CONSTRUCTIVE FORM OF A REFUSAL** (`.wiki/palo/icb.md` §3): the
    /// binding recipe is derived by walking one template against several
    /// synthetic descriptors and fitting each moving component to
    /// `v = base + Σ slope·axis`. A component that moves and does not fit is
    /// named here rather than guessed at, because a wrong slope is a grid
    /// that reads past a rectangle and no test between here and the tokens
    /// would say so.
    Unaffine {
        /// The slot, in walk order — which is dispatch order and ICB index.
        slot: u32,
        /// The shader point standing in that slot.
        point: String,
        /// Which component of it: a grid axis, a threadgroup axis, or an
        /// argument index.
        at: String,
        /// What the fit saw.
        why: String,
    },

    /// Two synthetic descriptors did not walk the same template.
    ///
    /// The claim design §5 makes — one artifact, all compositions inside it —
    /// stated as a check: the slots have to be the same slots in the same
    /// order, binding the same reservations at the same argument indices, or
    /// no single indirect command buffer serves both compositions and the
    /// exec key has not collapsed.
    Unstructured {
        /// Where the two recordings first disagree.
        slot: u32,
        /// How.
        why: String,
    },

    /// The plan names something this shell bound no seat for.
    Unbound {
        /// What went unbound.
        what: String,
    },

    /// **A residency this shell cannot arrange** (alto design §7,
    /// `crate::experts`).
    ///
    /// Every sentence in it names both numbers, because there is exactly one
    /// deployment action behind each: a `device_weight_budget` under the dense
    /// planes, a budget under one expert seat, a capped budget over a plan
    /// with nothing routed to hold less of, a segment routing to more distinct
    /// experts than its slab seats, or a bake whose regions carry two
    /// mixtures each.
    ///
    /// Carries a sentence rather than fields for the reason the constitution
    /// gives a refusal: the five shapes share no arithmetic, and a struct
    /// wide enough for all of them would print an empty half at every site.
    /// It lifts to `Error::Impossible` and never to `Exhausted` — nothing the
    /// deployment frees changes the answer.
    Residency(String),

    // attn-score — appended, so every prior variant keeps its ordinal.
    /// A capturing lane in a fire whose loaded artifact declares no capture
    /// arm (`.wiki/alto/attn-score.md` §4, palo C4).
    ///
    /// [`Fault::Maskless`]'s and [`Fault::Adapterless`]'s twin, one axis over
    /// and for their reason: the observability column is a fact the MODEL
    /// declares, so a plan with no `attn.scores` export has nowhere for the
    /// mass to go, and answering `Ok` would hand the caller an empty capture
    /// it could not tell from a captured nothing.
    Scoreless {
        /// The lane, in submission order.
        lane: u32,
    },

    /// A lane whose fact word and whose capture ask do not agree, in either
    /// direction.
    ///
    /// [`Fault::MaskWord`]'s and [`Fault::AdapterWord`]'s twin, and the same
    /// sentence: the word chooses the class and the class chooses whether this
    /// lane's rows fall inside the CAPTURE window. A capturing word with no
    /// ask behind it writes a plane the shell bound no epilogue at; an ask
    /// with a plain word lands the lane on a kernel that produces no mass at
    /// all, and the caller is handed a row of zeros that reads as "this lane
    /// attended to nothing".
    ScoreWord {
        /// The lane, in submission order.
        lane: u32,
        /// The word the caller stamped.
        word: u64,
        /// Whether the word's class runs the capture arm.
        runs_capture_arm: bool,
    },
}

impl Fault {
    /// One guest-program refusal, named where it happened.
    ///
    /// The constructor the program plane reaches for forty times over, so
    /// the two fields are never spelled at a call site.
    #[must_use]
    pub(crate) fn program(at: &'static str, why: impl Into<String>) -> Fault {
        Fault::Program {
            at,
            why: why.into(),
        }
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deviceless => write!(
                f,
                "this build has no Metal in it: the target is not an Apple one, so the \
                 device half is its refusing twin"
            ),
            Self::Device { call, why } => write!(f, "`{call}` refused: {why}"),
            Self::Shader {
                file,
                entrypoint,
                why,
            } => write!(f, "`{entrypoint}` of `{file}`: {why}"),
            Self::Ceiling { what, need, have } => write!(
                f,
                "this fire asks for {need} {what} and the load reserved {have}"
            ),
            Self::Bake(refusal) => write!(f, "the plan does not bake: {refusal}"),
            Self::Load(error) => write!(f, "the checkpoint does not land: {error}"),
            Self::Fire(error) => write!(f, "the fire is refused: {error}"),
            Self::Param { name, why } => write!(f, "param `{name}`: {why}"),
            Self::Fragmented {
                region,
                runs,
                promised,
            } => match promised {
                None => write!(
                    f,
                    "region {region} covers {runs} runs of the fire's rows and P4 wrote \
                     it no fallback row — it seriated so that this mask takes exactly \
                     one encode, and this fire's class order did not come from that \
                     seriation"
                ),
                Some(promised) => write!(
                    f,
                    "region {region} covers {runs} runs of the fire's rows where P4 \
                     counted {promised} on the order it shipped — a fire's order is \
                     that order with the absent classes dropped, and dropping a class \
                     cannot open a gap"
                ),
            },
            Self::Straddled {
                value,
                node,
                planned,
                consumed,
            } => write!(
                f,
                "value {value} is planned for classes {planned} and node {node} reads it \
                 under {consumed}; mint a second schedule"
            ),
            Self::Mask {
                lane,
                stated,
                extent,
            } => write!(
                f,
                "lane {lane} states a mask over {stated} keys and will read {extent}; a \
                 mask must REACH the extent (a longer one is fine and is clipped)"
            ),
            Self::Maskless { lane } => write!(
                f,
                "lane {lane} carries a mask and this artifact bakes no masked class"
            ),
            Self::MaskRows {
                lane,
                stated,
                rows,
            } => write!(
                f,
                "lane {lane} states a per-row attention mask (`Masking::Rows`) of \
                 {stated} rows and feeds {rows}; the form is parallel to the lane's \
                 tokens, and serving a short one as row zero's mask on every row is \
                 the substitution the form exists to end"
            ),
            Self::Positions {
                lane,
                stated,
                rows,
            } => write!(
                f,
                "lane {lane} states {stated} token positions and feeds {rows} tokens; a \
                 stated run is parallel to the lane's tokens or it is not stated at all"
            ),
            Self::MaskWord {
                lane,
                word,
                runs_masked_arm,
            } => write!(
                f,
                "lane {lane}'s word {word:#x} {} the masked arm and its mask says the \
                 other thing",
                if *runs_masked_arm { "runs" } else { "skips" }
            ),
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
            Self::Scoreless { lane } => write!(
                f,
                "lane {lane} asks to capture its attention mass and this load's artifact \
                 declares no capture arm: a score axis is a fact the MODEL states and \
                 this plan carries no `attn.scores` export to read"
            ),
            Self::ScoreWord {
                lane,
                word,
                runs_capture_arm,
            } => {
                if *runs_capture_arm {
                    write!(
                        f,
                        "lane {lane}'s word {word:#x} puts it in a class that writes a \
                         capture column, and it did not ask to be observed, so the mass \
                         would be computed into a plane no epilogue is pointed at"
                    )
                } else {
                    write!(
                        f,
                        "lane {lane} asks to capture its attention mass and its word \
                         {word:#x} puts it in a class outside the capture window, so no \
                         mass would be computed and a row of zeros would be \
                         indistinguishable from a capture of nothing"
                    )
                }
            }
            Self::Adapter { bank, why } => write!(f, "adapter bank `{bank}`: {why}"),
            Self::Compile(failure) => write!(
                f,
                "the guest program does not compile: {}",
                failure.reason()
            ),
            Self::Program { at, why } => write!(f, "`{at}` refused: {why}"),
            Self::Interpret(error) => write!(f, "{error}"),
            Self::Unaffine {
                slot,
                point,
                at,
                why,
            } => write!(
                f,
                "slot {slot} ({point}): {at} is not affine in the descriptor — {why}"
            ),
            Self::Unstructured { slot, why } => write!(
                f,
                "two compositions do not walk the same template at slot {slot}: {why}"
            ),
            Self::Unbound { what } => write!(
                f,
                "the plan names {what}, which this shell binds no seat for"
            ),
            Self::Residency(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Fault {}

impl From<model_compiler::Error> for Fault {
    fn from(refusal: model_compiler::Error) -> Self {
        Self::Bake(refusal)
    }
}

impl From<checkpoint::error::Error> for Fault {
    fn from(error: checkpoint::error::Error) -> Self {
        Self::Load(error)
    }
}

impl From<model_exec::Error> for Fault {
    fn from(error: model_exec::Error) -> Self {
        Self::Fire(error)
    }
}

impl From<model_exec::KernelError> for Fault {
    fn from(error: model_exec::KernelError) -> Self {
        Self::Fire(model_exec::Error::from(error))
    }
}

impl From<eta_exec::Failure> for Fault {
    fn from(failure: eta_exec::Failure) -> Self {
        Self::Compile(failure)
    }
}

impl From<eta_exec::Error> for Fault {
    fn from(error: eta_exec::Error) -> Self {
        Self::Interpret(error)
    }
}


/// What every fallible entry in this shell answers.
pub type Result<T> = std::result::Result<T, Fault>;

/// A kernel entry's refusal, reaching a shell path that answers [`Fault`].
///
/// **NOT EVERY CALL INTO `kernels-metal` IS A DISPATCH ARM.** Weight staging,
/// the wave's control launches and the fire's own scratch work all call
/// entries there and answer `Fault`, not the contract; before the shared
/// error crate came apart this was `From<KernelError>` and `?` did the work.
/// It still can, because `Fault` is this crate's own type and the orphan rule
/// only bites when NEITHER side is — which is the whole difference between
/// this impl and [`kernel`] below.
impl From<kernels_metal::Error> for Fault {
    fn from(error: kernels_metal::Error) -> Self {
        Fault::from(kernel(error))
    }
}

// ---------------------------------------------------------------------------
// the seam: this backend's refusal, said in the contract's words
// ---------------------------------------------------------------------------

/// Say a [`kernels_metal::Error`] in the dispatch contract's words.
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
/// in an inherent method that answers [`kernels_metal::Error`] so their own
/// `?` still converts.
///
/// **The match is total on purpose.** The two enums are variant for variant
/// identical today; they were one type until `crates/kernels` came apart, and
/// `model_exec::KernelError`'s own doc says plainly that three copies is a
/// prediction rather than a fact, with the falsifier written out. What makes
/// the prediction safe to hold is this function: the day `kernels-metal`
/// grows the `MTLLibrary` compile-failure variant it is expected to, this stops
/// compiling, at the one line that has to decide what the new refusal means
/// to a caller who can make nothing of a shader diagnostic. Nothing has to watch the
/// copy — and a copy that needs watching is the one
/// `crates/eta-exec/Cargo.toml` rules out: "a copy that is only safe because
/// something watches it is a copy that costs the watch".
pub fn kernel(error: kernels_metal::Error) -> KernelError {
    match error {
        kernels_metal::Error::Unsupported { op } => KernelError::Unsupported { op },
        kernels_metal::Error::DtypeUnsupported { op, dtype } => {
            KernelError::DtypeUnsupported { op, dtype }
        }
        kernels_metal::Error::Backend { op, detail } => KernelError::Backend { op, detail },
    }
}
