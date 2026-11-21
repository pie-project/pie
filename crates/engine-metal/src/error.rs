use std::fmt;

use poem_exec::KernelError;

#[derive(Debug)]
pub enum Fault {
    /// The disk tier's slot file is full ([`engine::Error::TierFull`]).
    TierFull(String),

    Deviceless,

    /// The system would not back `bytes` more: memory it is short of, not a
    /// load's arithmetic.
    Refused {
        bytes: u64,
    },

    Device {
        call: &'static str,
        why: String,
    },

    Shader {
        file: &'static str,
        entrypoint: &'static str,
        why: String,
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

    MaskUnread {
        lane: u32,
        word: u64,
    },

    /// The instance attached at `lane` describes, through its geometry
    /// ports, more lanes than its attachment owns: refused, never setting
    /// another request's positions or kv slots.
    GeometryLanes {
        lane: u32,
        instance: u64,
        carried: u32,
        owned: u32,
    },

    /// The instance attached at `lane` reads an output's rows past what its
    /// lanes hold: refused, never launched into another request's rows.
    OutputRows {
        lane: u32,
        instance: u64,
        why: String,
    },

    Adapterless {
        lane: u32,
    },

    AdapterWord {
        lane: u32,
        word: u64,
        runs_correction: bool,
    },

    Compile(eta_exec::Failure),

    /// Two live guest holders' state would overlap, or a member alone would
    /// move its group's pair: a bind or a rebind refused before it wrote
    /// there ([`engine::guest_claims`]).
    Claimed(engine::guest_claims::Refusal),

    Program {
        at: &'static str,
        why: String,
    },

    Interpret(eta_exec::Error),

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

    Unaffine {
        slot: u32,
        point: String,
        at: String,
        why: String,
    },

    Unstructured {
        slot: u32,
        why: String,
    },

    Unbound {
        what: String,
    },

    Residency(String),

    /// A fire met a pipeline the load did not build before ready.
    Unprepared {
        file: &'static str,
        entrypoint: &'static str,
        stamp: &'static str,
    },
    /// The state pool does not seat one working set beside the widest
    /// fire's activation under the device's allowance: the load's fit.
    FitShort(engine::chunk::FitShort),

    Backing {
        step: &'static str,
        bytes: u64,
        why: String,
    },

    Mapped {
        step: &'static str,
        what: String,
        why: String,
    },

    Recipe(String),

    /// A fire names a state slot whose chunk lies inside the activation
    /// span's backed carve: the activation, the guests' tables and their
    /// scratch lie there, so reading or writing the slot would read or
    /// overwrite them ([`crate::store::pool::Pool::clear_of_carve`]).
    UnderCarve {
        logical: u32,
        physical: u32,
        carve: u64,
        range: u64,
    },
}

impl Fault {
    #[must_use]
    pub(crate) fn program(at: &'static str, why: impl Into<String>) -> Fault {
        Fault::Program {
            at,
            why: why.into(),
        }
    }
}

impl Fault {
    /// The lane of its step a fault refuses for its own sake, where one
    /// does: what the lane states (its word, mask, positions, adapter,
    /// payload) is what this load cannot serve, and every other lane of
    /// the step could be ([`engine::Error::Lanes`]).
    #[must_use]
    pub fn lane(&self) -> Option<u32> {
        match self {
            Self::PatchPayload { lane, .. }
            | Self::Mask { lane, .. }
            | Self::Maskless { lane, .. }
            | Self::MaskRows { lane, .. }
            | Self::Positions { lane, .. }
            | Self::MaskUnread { lane, .. }
            | Self::GeometryLanes { lane, .. }
            | Self::OutputRows { lane, .. }
            | Self::Adapterless { lane, .. }
            | Self::AdapterWord { lane, .. } => Some(*lane),
            Self::Fire(error) => error.lane(),
            _ => None,
        }
    }

    /// A mask staging's fault, its lane counted in `rows`' order, named by
    /// the step's lane that row is of.
    #[must_use]
    pub fn of_rows(mut self, rows: &[poem_exec::fire::LaneRow]) -> Fault {
        if let Self::Mask { lane, .. }
            | Self::MaskRows { lane, .. } = &mut self
            && let Some(row) = rows.get(*lane as usize)
        {
            *lane = row.source;
        }
        self
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused { bytes } => {
                write!(f, "the system would not back {bytes} bytes more")
            }
            Self::Deviceless => write!(
                f,
                "this build has no Metal in it: the target is not an Apple one, so the \
                 device half is its refusing twin"
            ),
            Self::Device { call, why } => write!(f, "`{call}` refused: {why}"),
            Self::TierFull(why) => write!(f, "the disk tier's slot file is full: {why}"),
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
            Self::GeometryLanes {
                lane,
                instance,
                carried,
                owned,
            } => write!(
                f,
                "the instance {instance} attached at lane {lane} is refused: its \
                 `embed_indptr` port describes {carried} lane(s) and its attachment owns \
                 {owned}; past them its ports would set another request's positions and kv \
                 slots"
            ),
            Self::OutputRows {
                lane,
                instance,
                why,
            } => write!(
                f,
                "the instance {instance} attached at lane {lane} is refused: {why}; it reads \
                 only the rows its own lanes hold"
            ),
            Self::MaskUnread { lane, word } => write!(
                f,
                "lane {lane} states an attention mask and its word {word:#x} reads it in no \
                 arm this engine applies a mask in: here a selection's and a bias's arms \
                 read no per-lane mask"
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
            Self::Recipe(refusal) => f.write_str(refusal),
            Self::UnderCarve {
                logical,
                physical,
                carve,
                range,
            } => write!(
                f,
                "state chunk {logical} (physical {physical}) lies inside the activation \
                 span's backed carve, the top {carve} bytes of the {range}-byte pool, where a \
                 fire lays its activation and its guests' tables: refused, not read or written"
            ),
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
            Self::Claimed(refusal) => write!(f, "`guest::claims` refused: {refusal}"),
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
            Self::Unprepared { file, entrypoint, stamp } => write!(
                f,
                "this fire meets pipeline `{entrypoint}` of `{file}` ({stamp}), which the load \
                 did not build before ready: a load builds every pipeline its rungs dispatch, so \
                 this key is one the load's walk does not reach, and the fire is refused rather \
                 than compiled while serving"
            ),
            Self::FitShort(short) => write!(f, "{short}"),
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

impl From<engine::ports::PortError> for Fault {
    fn from(error: engine::ports::PortError) -> Self {
        Self::program("program::ports", error.0)
    }
}

pub type Result<T> = std::result::Result<T, Fault>;

impl From<kernels_metal::Error> for Fault {
    fn from(error: kernels_metal::Error) -> Self {
        Fault::from(kernel(error))
    }
}

pub fn kernel(error: kernels_metal::Error) -> KernelError {
    match error {
        kernels_metal::Error::Unsupported { op } => KernelError::Unsupported { op },
        kernels_metal::Error::DtypeUnsupported { op, dtype } => {
            KernelError::DtypeUnsupported { op, dtype }
        }
        kernels_metal::Error::Backend { op, detail } => KernelError::Backend { op, detail },
    }
}

impl From<engine::guest_claims::Overlap> for Fault {
    fn from(overlap: engine::guest_claims::Overlap) -> Fault {
        Fault::Claimed(overlap.into())
    }
}

impl From<engine::guest_claims::Refusal> for Fault {
    fn from(refusal: engine::guest_claims::Refusal) -> Fault {
        Fault::Claimed(refusal)
    }
}
