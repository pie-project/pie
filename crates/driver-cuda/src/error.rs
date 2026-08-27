//! What the shell refuses, and whose fault it is.
//!
//! THREE ERROR VOCABULARIES MEET HERE AND NONE OF THEM ABSORBS ANOTHER.
//! [`kernels::KernelError`] is about the backend and never about the plan;
//! [`driver::Error`] is about a fire the artifact cannot describe; and this
//! type is about everything that happens *around* a fire — binding a device,
//! landing a checkpoint, carving bytes. Folding any of the three into another
//! would send an operator hunting for a missing kernel when what actually
//! failed was a `cudaMalloc`, which is the failure mode the split exists to
//! prevent.
//!
//! An integrity break — a weight row the shell never bound, a cache id in a
//! tensor seat — is NOT here: [`Run`](crate::run::Run) panics with a sentence
//! for those, because they are bugs in this crate rather than conditions a
//! caller can act on. What reaches this enum is always something the caller
//! did, or something the machine did.

use std::fmt;

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
    Bake(model_compiler::Refusal),

    /// The loader refused to land this checkpoint.
    Load(model_loader::error::Error),

    /// The fire substrate refused this batch, or the backend refused a
    /// dispatch inside it.
    Fire(driver::Error),

    /// A region whose classes this fire's order does not make consecutive.
    ///
    /// **P4's PROMISE, FOUND BROKEN.** The layout pass solves one global C1P
    /// instance so that every windowed consumer's class set is an interval of
    /// the class order (design §3), which is what lets a windowed kernel take
    /// a pointer and an extent. A region that comes back as more than one run
    /// would need more than one launch — a `Fallback` row — and the catalog
    /// bakes an empty [`FallbackTable`](model_compiler::FallbackTable) today,
    /// so there is nothing to fall back to. It is a bake-integrity failure,
    /// refused by name rather than run over the classes in between.
    Fragmented {
        /// Which region of `Baked::template`.
        region: u32,
        /// How many runs its mask covers.
        runs: usize,
    },

    /// A tensor the checkpoint published that this plan does not name, or a
    /// plan param the checkpoint never published.
    Param {
        /// The name, as the plan and the load contract both spell it.
        name: String,
        /// Which side was missing it.
        why: &'static str,
    },

    /// A count past a ceiling the shell reserved bytes for.
    Ceiling {
        /// What overflowed.
        what: &'static str,
        /// What was asked.
        need: u64,
        /// What was reserved.
        have: u64,
    },

    /// A fire whose schedules are not the shape the graph for its key was
    /// captured against.
    ///
    /// **THE ONE CLAIM A GRAPH KEY CANNOT CHECK BY ITSELF.** A recorded fire
    /// bakes the attention schedule's offsets, its padded batch size and its
    /// tile width into the launches it recorded, and the prepare phase
    /// rebuilds that schedule every fire. Under
    /// [`FireBindings::capture`](crate::FireBindings) the builders carve
    /// graph-shaped schedules, so those numbers depend on the fire's SHAPE
    /// and the key holds the shape fixed — but that is somebody else's
    /// arithmetic, and a change in it would otherwise present as slightly
    /// wrong logits forever. Refused by name instead.
    Schedule {
        /// The shape key, as `record::Key` spells it.
        key: String,
    },

    /// A lane's mask does not describe the lane.
    ///
    /// **THE SHORT DIRECTION IS THE DANGEROUS ONE.** A mask states `total`
    /// positions and the lane's readable extent after this fire's append is
    /// `held + rows`; a mask that covers fewer positions than that would
    /// expand into a rectangle whose tail bits are zero, and zero is
    /// MASKED-OUT — a silently truncated attention rather than a fault. A
    /// longer one is somebody else's mask. Neither is repaired.
    Mask {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// How many positions the mask says it covers.
        stated: u64,
        /// How many the lane will hold once this fire's tokens are written.
        extent: u64,
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
    /// vocabulary note). `masked` is a bit in the word the engine stamps from
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
    /// whole fire and no per-lane state, so the engine stamps `masked` on
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
        /// The plan value, as `Plan::values` numbers it.
        value: u32,
        /// The node consuming it, as `Plan::nodes` numbers it.
        node: u32,
        /// The classes its defining region runs in.
        planned: String,
        /// The classes the consuming region runs in.
        consumed: String,
    },

    /// A guest program (PTIR) that does not compile on this device.
    ///
    /// **THE TAXONOMY IS THE POINT, NOT THE TEXT.**
    /// [`Deterministic`](driver::Failure::Deterministic) means the source is
    /// wrong and will be wrong next time — the compile plane remembers it and
    /// answers the next registration from memory. `Retryable` means the
    /// machine could not, this time (no NVRTC, out of memory, a cubin that
    /// would not load), and remembering it would strand a program on one bad
    /// minute. Folding the two into one string is what makes a driver either
    /// retry a syntax error forever or give up on a transient.
    Compile(driver::Failure),

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

    /// A plan naming something this shell has no binding for.
    ///
    /// A refusal rather than a panic because it is a statement about the
    /// PLAN — a model this shell cannot serve yet — and the caller's recovery
    /// is to load a different one.
    Unbound {
        /// The seat, named as the IR names it.
        what: String,
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
            Self::Load(error) => write!(f, "this checkpoint does not land: {error}"),
            Self::Fire(error) => write!(f, "{error}"),
            Self::Fragmented { region, runs } => write!(
                f,
                "region {region} covers {runs} runs of this fire's rows, and a \
                 windowed launch takes one pointer and one extent — P4 seriates \
                 so that it takes exactly one"
            ),
            Self::Param { name, why } => write!(f, "`{name}` {why}"),
            Self::Ceiling { what, need, have } => write!(
                f,
                "this fire wants {need} {what} and the shell reserved {have}"
            ),
            Self::Schedule { key } => write!(
                f,
                "the attention schedules this fire built are not the shapes the graph \
                 for {key} was captured against, and a replay would read the captured \
                 ones"
            ),
            Self::Mask {
                lane,
                stated,
                extent,
            } => write!(
                f,
                "lane {lane}'s mask covers {stated} positions and the lane will hold \
                 {extent} once this fire's tokens are written; a mask is over the \
                 lane's whole readable extent, and a short one masks out the tail \
                 rather than leaving it alone"
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
            Self::Unbound { what } => write!(
                f,
                "this plan names {what}, which this shell does not bind"
            ),
        }
    }
}

impl std::error::Error for Fault {}

impl From<model_compiler::Refusal> for Fault {
    fn from(refusal: model_compiler::Refusal) -> Fault {
        Fault::Bake(refusal)
    }
}

impl From<model_loader::error::Error> for Fault {
    fn from(error: model_loader::error::Error) -> Fault {
        Fault::Load(error)
    }
}

impl From<driver::Error> for Fault {
    fn from(error: driver::Error) -> Fault {
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

impl From<driver::Failure> for Fault {
    fn from(failure: driver::Failure) -> Fault {
        Fault::Compile(failure)
    }
}

impl From<kernels::KernelError> for Fault {
    fn from(error: kernels::KernelError) -> Fault {
        Fault::Fire(driver::Error::Kernel(error))
    }
}
