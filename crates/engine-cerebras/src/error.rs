use std::fmt;

use poem_exec::KernelError;

#[derive(Debug)]
pub enum Fault {
    Deviceless,

    /// The CSL compiler or the runner refused a program.
    Compile {
        why: String,
    },

    NoDevice {
        detail: String,
    },

    Ceiling {
        what: &'static str,

        need: u64,

        have: u64,
    },

    PatchPayload {
        lane: u32,

        what: &'static str,

        have: u64,

        want: u64,
    },

    Bake(poem_compiler::Error),

    Load(checkpoint::error::Error),

    Fire(poem_exec::Error),

    Param {
        name: String,

        why: &'static str,
    },

    Fragmented {
        region: u32,

        runs: usize,

        promised: Option<u32>,
    },

    Straddled {
        value: u32,

        node: u32,

        planned: String,

        consumed: String,
    },

    Mask {
        lane: u32,

        stated: u64,

        extent: u64,
    },

    Maskless {
        lane: u32,
    },

    MaskRows {
        lane: u32,

        stated: u64,

        rows: u32,
    },

    Positions {
        lane: u32,

        stated: u64,

        rows: u64,
    },

    MaskWord {
        lane: u32,

        word: u64,

        runs_masked_arm: bool,
    },

    Adapterless {
        lane: u32,
    },

    Program {
        at: &'static str,

        why: String,
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

    Unbound {
        what: String,
    },

    Residency(String),

    Recipe(String),
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deviceless => write!(f, "no device is open on this engine"),
            Self::Compile { why } => write!(f, "the program did not compile or run: {why}"),
            Self::NoDevice { detail } => write!(f, "no Cerebras device to open: {detail}"),
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
            Self::MaskRows { lane, stated, rows } => write!(
                f,
                "lane {lane} states a per-row attention mask (`Masking::Rows`) of \
                 {stated} rows and feeds {rows}; the form is parallel to the lane's \
                 tokens, and serving a short one as row zero's mask on every row is \
                 the substitution the form exists to end"
            ),
            Self::Positions { lane, stated, rows } => write!(
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
            Self::Program { at, why } => write!(f, "`{at}` refused: {why}"),
            Self::Unbound { what } => write!(
                f,
                "the plan names {what}, which this shell binds no seat for"
            ),
            Self::Residency(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Fault {}

impl From<poem_compiler::Error> for Fault {
    fn from(refusal: poem_compiler::Error) -> Self {
        Self::Bake(refusal)
    }
}

impl From<checkpoint::error::Error> for Fault {
    fn from(error: checkpoint::error::Error) -> Self {
        Self::Load(error)
    }
}

impl From<poem_exec::Error> for Fault {
    fn from(error: poem_exec::Error) -> Self {
        Self::Fire(error)
    }
}

impl From<poem_exec::KernelError> for Fault {
    fn from(error: poem_exec::KernelError) -> Self {
        Self::Fire(poem_exec::Error::from(error))
    }
}

pub type Result<T> = std::result::Result<T, Fault>;

impl From<kernels_cerebras::Error> for Fault {
    fn from(error: kernels_cerebras::Error) -> Self {
        Fault::from(kernel(error))
    }
}

pub fn kernel(error: kernels_cerebras::Error) -> KernelError {
    match error {
        kernels_cerebras::Error::Unsupported { op } => KernelError::Unsupported { op },
        kernels_cerebras::Error::DtypeUnsupported { op, dtype } => {
            KernelError::DtypeUnsupported { op, dtype }
        }
        kernels_cerebras::Error::Backend { op, detail } => KernelError::Backend { op, detail },
    }
}
