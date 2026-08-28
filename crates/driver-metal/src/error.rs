//! The shell's answer vocabulary: one enum, one `Result`, and the rule that
//! a refusal names what refused.
//!
//! The split this file keeps is `kernels::error`'s, restated one layer up: a
//! [`KernelError`] is about the BACKEND — a geometry no shader is stamped
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

use std::fmt;

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
    Load(model_loader::error::Error),

    /// The fire substrate refused the batch, or a dispatch refused the op.
    Fire(driver::Error),

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
    /// `driver::fire::fallback`). What is left here is the case where the bake
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

    /// A lane's stated mask does not cover the extent it will read.
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

    /// A lane's word and its mask disagree, in either direction.
    MaskWord {
        /// The lane, in submission order.
        lane: u32,
        /// The word the caller stamped.
        word: u64,
        /// Whether the word's class runs the masked arm.
        runs_masked_arm: bool,
    },

    /// A guest program this device would not compile.
    ///
    /// Carries `driver::Failure`'s own split — a `Deterministic` refusal is
    /// remembered forever (the source will not compile on this device, ever)
    /// and a `Retryable` one is not (the machine was out of something). On
    /// this plane the split comes off `MTLLibraryError`: a compile failure
    /// is the source's, and anything else is the moment's.
    Compile(driver::Failure),

    /// The guest-program plane refused a call, and this is where and why.
    Program {
        /// Which entry refused, as the plane spells it.
        at: &'static str,
        /// The sentence.
        why: String,
    },

    /// An adapter registration this load's banks cannot seat.
    ///
    /// Reachable even though `linear.lora_correct` is `Unsupported` on this
    /// plane: a bank is a `ParamSource::Registered` weight, so
    /// [`Weights`](crate::Weights) reserves and zeroes one for any plan that
    /// declares it and `register_adapter` can be called against it. What
    /// cannot happen yet is the CORRECTION — the op refuses by name at its
    /// first node — so a caller that registered planes here has a residency
    /// that nothing reads. That is a truthful state and not an error, which
    /// is why this variant is about the registration and not about the axis.
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
                "lane {lane} states a mask over {stated} keys and will read {extent}"
            ),
            Self::Maskless { lane } => write!(
                f,
                "lane {lane} carries a mask and this artifact bakes no masked class"
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
            Self::Adapter { bank, why } => write!(f, "adapter bank `{bank}`: {why}"),
            Self::Compile(failure) => write!(
                f,
                "the guest program does not compile: {}",
                failure.reason()
            ),
            Self::Program { at, why } => write!(f, "`{at}` refused: {why}"),
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
        }
    }
}

impl std::error::Error for Fault {}

impl From<model_compiler::Error> for Fault {
    fn from(refusal: model_compiler::Error) -> Self {
        Self::Bake(refusal)
    }
}

impl From<model_loader::error::Error> for Fault {
    fn from(error: model_loader::error::Error) -> Self {
        Self::Load(error)
    }
}

impl From<driver::Error> for Fault {
    fn from(error: driver::Error) -> Self {
        Self::Fire(error)
    }
}

impl From<kernels::KernelError> for Fault {
    fn from(error: kernels::KernelError) -> Self {
        Self::Fire(driver::Error::from(error))
    }
}

impl From<driver::Failure> for Fault {
    fn from(failure: driver::Failure) -> Self {
        Self::Compile(failure)
    }
}


/// What every fallible entry in this shell answers.
pub type Result<T> = std::result::Result<T, Fault>;
