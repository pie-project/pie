use std::fmt;

use model_exec::KernelError;

pub type Result<T> = std::result::Result<T, Fault>;

#[derive(Debug)]
pub enum Fault {
    Runtimeless,

    Device {
        call: &'static str,
        code: i32,
    },

    Bake(model_compiler::Error),

    Load(checkpoint::error::Error),

    Fire(model_exec::Error),

    Fragmented {
        region: u32,
        runs: usize,
        promised: Option<u32>,
    },

    Param {
        name: String,
        why: &'static str,
    },

    Residency(String),

    PatchPayload {
        lane: u32,
        need: u64,
        have: u64,
    },
    VoxelPayload {
        lane: u32,
        what: &'static str,
    },

    Ceiling {
        what: &'static str,
        need: u64,
        have: u64,
    },

    Mask {
        lane: u32,
        stated: u64,
        extent: u64,
    },

    MaskRows {
        lane: u32,
        stated: u64,
        rows: u32,
    },

    Maskless {
        lane: u32,
    },

    MaskWord {
        lane: u32,
        word: u64,
        runs_masked_arm: bool,
    },

    Adapterless {
        lane: u32,
    },

    AdapterWord {
        lane: u32,
        word: u64,
        runs_correction: bool,
    },

    OutOfMemory {
        need: u64,
        have: u64,
    },

    Draftless {
        lane: u32,
    },

    DraftWord {
        lane: u32,
        word: u64,
        runs_draft_arm: bool,
    },

    Scoreless {
        lane: u32,
    },

    ScoreWord {
        lane: u32,
        word: u64,
        runs_capture_arm: bool,
    },

    Adapter {
        bank: String,
        why: String,
    },

    Blob {
        path: String,
        why: String,
    },

    AdapterSlots {
        seats: u32,
    },

    Straddled {
        value: u32,
        node: u32,
        planned: String,
        consumed: String,
    },

    Compile(eta_exec::Failure),

    Program {
        at: &'static str,
        why: String,
    },

    Integrity {
        at: &'static str,
        why: String,
    },

    Interpret(eta_exec::Error),

    Unbound {
        what: String,
    },

    Unlowered {
        region: u32,
        lowering: String,
    },

    Golden {
        key: String,
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
            Self::VoxelPayload { lane, what } => {
                write!(
                    f,
                    "lane {lane} submitted clips this fire cannot seat: {what}"
                )
            }
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
            Self::MaskRows { lane, stated, rows } => write!(
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
            Self::Unbound { what } => {
                write!(f, "this plan names {what}, which this shell does not bind")
            }
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

impl From<kernels_cuda::Error> for Fault {
    fn from(error: kernels_cuda::Error) -> Self {
        Fault::from(kernel(error))
    }
}

pub fn kernel(error: kernels_cuda::Error) -> KernelError {
    match error {
        kernels_cuda::Error::Unsupported { op } => KernelError::Unsupported { op },
        kernels_cuda::Error::DtypeUnsupported { op, dtype } => {
            KernelError::DtypeUnsupported { op, dtype }
        }
        kernels_cuda::Error::Backend { op, detail } => KernelError::Backend { op, detail },
    }
}
