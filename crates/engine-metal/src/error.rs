//! The shell's error vocabulary: [`KernelError`] is about the backend,
//! [`Fault`] is about the shell.

use std::fmt;

use model_exec::KernelError;

/// Every way this shell answers "no".
#[derive(Debug)]
pub enum Fault {
    /// No Metal in this build — a non-Apple target.
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

    /// A lane's media geometry and media payload disagree — a check
    /// `engine::fire::StepMedia::validate` cannot make since it needs shell-
    /// only numbers (patch row width, position-table tap count).
    PatchPayload {
        /// Which lane of the submission.
        lane: u32,
        /// Which of its vectors.
        what: &'static str,
        /// What that vector carries.
        have: u64,
        /// What its own geometry owes.
        want: u64,
    },

    /// The compiler refused to bake the plan against these budgets.
    Bake(model_compiler::Error),

    /// The loader refused to land the checkpoint against this plan.
    Load(checkpoint::error::Error),

    /// The fire substrate refused the batch, or a dispatch refused the op.
    Fire(model_exec::Error),

    /// A param the plan names that the checkpoint never published, or
    /// published at another shape.
    Param {
        /// The param's canonical name.
        name: String,
        /// Why it could not land.
        why: &'static str,
    },

    /// A region's classes are not one run and the artifact owes it no
    /// fallback: the bake and the fire disagree about the compiler's
    /// consecutiveness promise.
    Fragmented {
        /// The template region.
        region: u32,
        /// How many runs its class mask fell into in this fire.
        runs: usize,
        /// How many the compiler wrote down, or `None` when it wrote nothing at all —
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

    /// A lane's stated mask does not reach the extent it will read (a longer
    /// mask is fine and gets clipped).
    Mask {
        /// The lane, in submission order.
        lane: u32,
        /// The mask's stated total.
        stated: u64,
        /// The extent the fire will read.
        extent: u64,
    },

    /// A lane carries a mask and this artifact bakes no masked class.
    Maskless {
        /// The lane, in submission order.
        lane: u32,
    },

    /// A lane's per-row mask (`Masking::Rows`) states a different number of
    /// rows than the lane feeds.
    MaskRows {
        /// The lane, in submission order.
        lane: u32,
        /// How many rows the masking states.
        stated: u64,
        /// How many rows the lane feeds.
        rows: u32,
    },

    /// A lane states its own token positions and a different number of them
    /// than it feeds tokens. Monotonicity is deliberately not checked here —
    /// a non-monotone run is a caller's rotation, not a geometry error.
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

    /// An adapted lane in a fire whose loaded artifact bakes no correction
    /// arm, so it would get the base model's answer under an adapter's name.
    Adapterless {
        /// The lane, in submission order.
        lane: u32,
    },

    /// A lane whose fact word and whose adapter disagree: the word's class
    /// decides whether the lane's rows fall inside the correction's window.
    AdapterWord {
        /// The lane, in submission order.
        lane: u32,
        /// The word the caller stamped.
        word: u64,
        /// Whether the word's class runs the correction.
        runs_correction: bool,
    },

    /// A guest program this device would not compile: `eta_exec::Failure`'s
    /// `Deterministic` vs `Retryable` split.
    Compile(eta_exec::Failure),

    /// The guest-program plane refused a call, and this is where and why.
    Program {
        /// Which entry refused, as the plane spells it.
        at: &'static str,
        /// The sentence.
        why: String,
    },

    /// The ETA substrate refused a launch program, distinct from
    /// [`Fault::Program`] (this crate's own guest-program refusals). Both
    /// sort to `Error::Program` at the contract.
    Interpret(eta_exec::Error),

    /// An adapter registration this load's banks cannot seat: an undeclared
    /// bank name, an id past capacity, or a plane that is not one whole slot.
    Adapter {
        /// The bank, as the plan's param names it.
        bank: String,
        /// Why the planes do not fit it.
        why: String,
    },

    /// A shared adapter the mount, manifest or model text refuses — missing
    /// file, directory, or plane mismatch.
    Blob {
        /// The adapter, as the bind spelled it.
        path: String,
        /// What is wrong with it.
        why: String,
    },

    /// Every adapter slot this load's banks seat is pinned by a live bind.
    /// A refusal, not an eviction: a slot in flight is never taken back.
    AdapterSlots {
        /// How many slots the banks seat.
        seats: u32,
    },

    /// A quantity the ICB would rewrite per fire, and no affine law
    /// (`v = base + sum(slope*axis)`) predicts it.
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

    /// Two synthetic descriptors did not walk the same template (slots,
    /// order, argument indices), so no single ICB can serve both.
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

    /// A residency this shell cannot arrange (`crate::experts`). A
    /// sentence, not fields — the shapes share no arithmetic. Lifts to
    /// `Error::Impossible`, never `Exhausted` (freeing memory doesn't help).
    Residency(String),

    /// A capturing lane in a fire whose loaded artifact declares no capture
    /// arm.
    Scoreless {
        /// The lane, in submission order.
        lane: u32,
    },

    /// A lane whose fact word and whose capture ask disagree: the word's
    /// class decides whether the lane's rows fall inside the capture window.
    ScoreWord {
        /// The lane, in submission order.
        lane: u32,
        /// The word the caller stamped.
        word: u64,
        /// Whether the word's class runs the capture arm.
        runs_capture_arm: bool,
    },

    /// A streamed load whose backing temp-file mapping failed. Named
    /// separately from [`Fault::Load`]: the checkpoint is innocent here.
    Backing {
        /// Which call refused: `open`, `size` or `map`.
        step: &'static str,
        /// How many bytes of streamed source the plan asked to back.
        bytes: u64,
        /// The OS's own sentence.
        why: String,
    },

    /// An artifact this shell could not map, could not bind zero-copy, or
    /// was asked to write through — a mapped reservation is `PROT_READ`.
    Mapped {
        /// Which call refused — `open`, `stat`, `size`, `map`, `bind` — or
        /// the method a read-only reservation refused: `write`, `zero_span`.
        step: &'static str,
        /// The artifact, as the caller named it.
        what: String,
        /// The OS's, Metal's, or this shell's own sentence.
        why: String,
    },

    /// A serving artifact this deployment is not the one for — the
    /// `pie.serving/1` stamp gate, refused before any device byte is
    /// reserved. A sentence, not fields; shared wording with the CUDA shell.
    Recipe(String),
}

impl Fault {
    /// One guest-program refusal, named where it happened.
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
            Self::PatchPayload {
                lane,
                what,
                have,
                want,
            } => write!(
                f,
                "lane {lane}'s media carries {have} {what} and its own geometry owes \
                 {want} — its spans and its payload disagree"
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
                 and this plan has no `linear.lora_correct` arm for the id \
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
            Self::Backing { step, bytes, why } => write!(
                f,
                "this load streams {bytes} bytes of routed experts and the temporary file \
                 that would hold them does not `{step}`: {why}"
            ),
            Self::Mapped { step, what, why } => write!(
                f,
                "the artifact `{what}` is served from its own mapped pages and does not \
                 `{step}`: {why}"
            ),
            // Forwarded whole: the string already names its own subject.
            Self::Recipe(refusal) => f.write_str(refusal),
            Self::Adapter { bank, why } => write!(f, "adapter bank `{bank}`: {why}"),
            Self::Blob { path, why } => write!(f, "the shared adapter `{path}` {why}"),
            Self::AdapterSlots { seats } => write!(
                f,
                "this load's banks seat {seats} adapters and every slot is pinned by a \
                 live bind; a slot is never taken back from under one, because a fire \
                 reading it would answer another instance's adapter under this one's \
                 name. Close an instance that is done, or state a model text that seats \
                 more"
            ),
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

/// A kernel entry's refusal, reaching a shell path that answers [`Fault`]
/// directly (weight staging, control launches, scratch work) rather than
/// through the dispatch contract.
impl From<kernels_metal::Error> for Fault {
    fn from(error: kernels_metal::Error) -> Self {
        Fault::from(kernel(error))
    }
}

// The seam: this backend's refusal, said in the contract's words.

/// Say a [`kernels_metal::Error`] in the dispatch contract's words. A
/// function, not a `From` impl, since both types are foreign here (orphan
/// rule). The match is deliberately total so a new variant fails to compile
/// rather than being silently dropped.
pub fn kernel(error: kernels_metal::Error) -> KernelError {
    match error {
        kernels_metal::Error::Unsupported { op } => KernelError::Unsupported { op },
        kernels_metal::Error::DtypeUnsupported { op, dtype } => {
            KernelError::DtypeUnsupported { op, dtype }
        }
        kernels_metal::Error::Backend { op, detail } => KernelError::Backend { op, detail },
    }
}
