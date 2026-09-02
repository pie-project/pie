//! What the shell refuses, and whose fault it is.

use std::fmt;

use model_exec::KernelError;

/// The shell's result.
pub type Result<T> = std::result::Result<T, Fault>;

/// One refusal, naming what it was and the numbers behind it.
#[derive(Debug)]
pub enum Fault {
    /// This build selected no CUDA runtime.
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
    Fire(model_exec::Error),

    /// A region whose classes this fire's order does not make consecutive, and
    /// which the artifact owes no answer for.
    Fragmented {
        /// Which region of `CompiledModel::template`.
        region: u32,
        /// How many runs its mask covers in this fire.
        runs: usize,
        /// How many the fallback row promised, or `None` when it wrote nothing at all —
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

    /// A weight-residency budget this plan cannot be served under; not
    /// recoverable by freeing resources.
    Residency(String),

    /// A count past a ceiling the shell reserved bytes for.
    /// A media submission whose patch payload does not match the geometry
    /// beside it.
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

    /// A lane's mask does not reach the lane's readable extent. A short mask
    /// is refused (not zero-padded); a longer one is accepted and clipped.
    Mask {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// How many positions the mask says it covers.
        stated: u64,
        /// How many the lane will hold once this fire's tokens are written.
        extent: u64,
    },

    /// A lane's per-row mask does not have one mask per query row.
    MaskRows {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// How many per-row masks the lane stated.
        stated: u64,
        /// How many token rows this fire feeds the lane.
        rows: u32,
    },

    /// A masked lane in a fire whose loaded artifact bakes no masked class
    /// (no `attention.masked` arm to read it).
    Maskless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose mask do not agree.
    MaskWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the masked arm.
        runs_masked_arm: bool,
    },

    /// An adapted lane in a fire whose loaded artifact bakes no correction.
    Adapterless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose adapter do not agree.
    AdapterWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the correction.
        runs_correction: bool,
    },

    /// An allocation the device has no room for (unlike [`Fault::Ceiling`],
    /// which is a fire exceeding what a load reserved).
    OutOfMemory {
        /// Bytes asked for.
        need: u64,
        /// Bytes the device had free when the ask failed.
        have: u64,
    },

    /// A drafting lane in a fire whose loaded artifact declares no draft head.
    Draftless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose draft ask do not agree.
    DraftWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the draft head.
        runs_draft_arm: bool,
    },

    /// A capturing lane in a fire whose loaded artifact declares no capture
    /// arm.
    Scoreless {
        /// Which lane asked.
        lane: u32,
    },

    /// A lane whose fact word and whose capture ask do not agree.
    ScoreWord {
        /// Which lane of the fire, in submission order.
        lane: u32,
        /// The word it was stamped with.
        word: u64,
        /// Whether the class that word resolved to runs the capture arm.
        runs_capture_arm: bool,
    },

    /// A registration this load's banks cannot seat.
    Adapter {
        /// The bank the registration named.
        bank: String,
        /// What is wrong with it.
        why: String,
    },

    /// A shared adapter the mount cannot serve.
    Blob {
        /// The adapter, as the bind spelled it.
        path: String,
        /// What is wrong with it.
        why: String,
    },

    /// Every adapter slot pinned by a live bind when a load wanted one.
    /// `slots` bounds concurrent residency, not the catalog.
    AdapterSlots {
        /// How many slots the banks seat.
        seats: u32,
    },

    /// A plan struct built over more rows than the node consuming it runs.
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
    /// Deterministic failures are cached; retryable ones are not.
    Compile(eta_exec::Failure),

    /// The guest-program plane refused this call.
    Program {
        /// Which entry point refused, as this crate spells it.
        at: &'static str,
        /// The condition, in a sentence.
        why: String,
    },

    /// An invariant this crate owns, broken at serve time: a bug here, named
    /// rather than counted, so a deployment does not serve around it.
    Integrity {
        /// Where it was caught, as this crate spells it.
        at: &'static str,
        /// What was found, in a sentence.
        why: String,
    },

    /// The ETA substrate refused a launch program, in its own words.
    Interpret(eta_exec::Error),

    /// A plan naming something this shell has no binding for.
    Unbound {
        /// The seat, named as the IR names it.
        what: String,
    },

    /// A conditional bracket reaching a recording walk with nowhere to put
    /// it: a `Switch` arm with no row count, or a load whose context opened
    /// no body stream.
    Unlowered {
        /// Which region of the template.
        region: u32,
        /// The lowering, as the compiler spells it.
        lowering: String,
    },

    /// **A BODY THE LOAD ARMED ANSWERS SOMETHING OTHER THAN THE EAGER WALK
    /// OF ITS OWN COMPOSITION** — the golden's verdict (`[engine] golden`,
    /// `Shell::golden`), and it fails the load.
    ///
    /// **THIS IS A BUG IN THIS ENGINE AND NOT IN THE DEPLOYMENT**, which is
    /// why it is a refusal to boot and not a slower boot. Every other refusal
    /// on this enum is something the caller did or the machine did; this one
    /// is the engine caught computing a wrong answer for a shape it would
    /// have served. The honest answers were two: refuse the load, or seat
    /// the key on the eager road and serve the deployment slower. The second
    /// is a bypass — a body wrong at boot is a walk wrong somewhere, or a
    /// composition no caller can bring, and either is a fault this tree has
    /// to find — so the load fails, loudly, naming the key and the first
    /// differing element, and `[engine] golden = false` is the operator's
    /// override for a boot that must come up anyway.
    Golden {
        /// The `record::BodyKey`, as it prints.
        key: String,
        /// The lanes the key was armed with, as `Shell::golden` spells them
        /// (`word/rows` per lane), and the first difference measured.
        why: String,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtimeless => write!(
                f,
                "this build carries no CUDA runtime: enable `cuda`, \
                 matching the libcudart it will load"
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
                 and this plan has no `attention.masked` arm for the bits \
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
            Self::OutOfMemory { need, have } => write!(
                f,
                "this load wants {need} bytes of device memory and the device has \
                 {have} free"
            ),
            Self::Draftless { lane } => write!(
                f,
                "lane {lane} asks for the model's draft head and this load's artifact declares \
                 none: an MTP axis is a fact the MODEL states and this plan carries \
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
                 declares no capture arm: a score axis is a fact the MODEL states \
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
            Self::Integrity { at, why } => write!(f, "{at}: {why} (a fault in this engine)"),
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
            Self::Golden { key, why } => write!(
                f,
                "the golden refused this load: the body armed for {key} answers \
                 something other than the eager walk of its own composition ({why}). \
                 That is a fault in this engine and not in this deployment — the way \
                 in is to fire that key's synthetic both ways by hand — and \
                 `[engine] golden = false` boots without the check"
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
impl From<kernels_cuda::Error> for Fault {
    fn from(error: kernels_cuda::Error) -> Self {
        Fault::from(kernel(error))
    }
}

/// Says a [`kernels_cuda::Error`] in the dispatch contract's words. A free
/// function, not a `From` impl, since both types are foreign (orphan rule).
pub fn kernel(error: kernels_cuda::Error) -> KernelError {
    match error {
        kernels_cuda::Error::Unsupported { op } => KernelError::Unsupported { op },
        kernels_cuda::Error::DtypeUnsupported { op, dtype } => {
            KernelError::DtypeUnsupported { op, dtype }
        }
        kernels_cuda::Error::Backend { op, detail } => KernelError::Backend { op, detail },
    }
}
