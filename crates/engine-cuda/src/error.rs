use std::fmt;

use poem_exec::KernelError;

pub type Result<T> = std::result::Result<T, Fault>;

#[derive(Debug)]
pub enum Fault {
    Runtimeless,

    /// The disk tier's slot file is full ([`engine::Error::TierFull`]).
    TierFull(String),

    Device {
        call: &'static str,
        code: i32,
    },

    /// A device this engine does not serve, or one whose facts it could not
    /// read; never guessed at.
    Unsupported(String),

    Bake(poem_compiler::Error),

    Load(checkpoint::error::Error),

    Fire(poem_exec::Error),

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
    /// The card cannot hold what the load sized for it: the driver's own
    /// holdings beside the pool, or a range it refused.
    Card(String),

    Recipe(String),

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

    /// A fire's carve that asks a value more bytes than the arena map laid
    /// its slot: the value's rows would land in a neighbour's.
    Overcarved {
        value: u32,
        rows: u32,
        need: u64,
        laid: u64,
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

    MaskUnread {
        lane: u32,
        word: u64,
    },

    Adapterless {
        lane: u32,
    },

    /// A row the lane's program places lands outside the slots the lane may
    /// write (`engine::fire::misplaced_write`): refused before any of it runs.
    KvWrite {
        lane: u32,
        why: String,
    },

    /// A device-geometry program's `embed_indptr` port describes more lanes
    /// than its attachment spans: its ports would state the next lanes'
    /// positions and slots. Every lane of the attachment is refused, before
    /// any of it is written.
    Overreach {
        instance: engine::InstanceId,
        lanes: std::ops::Range<u32>,
        carried: usize,
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

    /// Two live guest holders' state would overlap, or a member alone would
    /// move its group's pair: a bind or a rebind refused before it wrote
    /// there ([`engine::guest_claims`]).
    Claimed(engine::guest_claims::Refusal),

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

    /// A generated region the skeletons do not take: the fold forms only what
    /// their `serves` admits, so this names the fold's bug.
    Region {
        node: u32,
        body: u64,
        why: String,
    },

    Golden {
        key: String,
        why: String,
    },
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
            | Self::MaskRows { lane, .. }
            | Self::Maskless { lane, .. }
            | Self::MaskUnread { lane, .. }
            | Self::Adapterless { lane, .. }
            | Self::KvWrite { lane, .. }
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
            Self::Runtimeless => write!(
                f,
                "this build carries no CUDA runtime: enable `cuda`, \
                 matching the libcudart it will load"
            ),
            Self::Device { call, code } => {
                write!(f, "{call} answered {code}")
            }
            Self::Unsupported(why) => write!(f, "this device is not served: {why}"),
            Self::Bake(refusal) => write!(f, "this plan does not bake: {refusal:?}"),
            Self::Residency(why) => write!(f, "weight residency: {why}"),
            Self::Card(why) => write!(f, "the card: {why}"),
            Self::Recipe(refusal) => f.write_str(refusal),
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
            Self::Overcarved {
                value,
                rows,
                need,
                laid,
            } => write!(
                f,
                "this fire carves value {value} at {rows} rows, {need} bytes, and the arena \
                 laid its slot for {laid}: its rows past the slot would land in another \
                 value's"
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
                 to reach, nor a ragged attention whose text states a lane's mask \
                 (`stated_group_mask()`)"
            ),
            Self::MaskUnread { lane, word } => write!(
                f,
                "lane {lane} states an attention mask and its word {word:#x} reads it in no \
                 arm this engine applies a mask in, so the mask would be staged and never \
                 read: here `attn.decode`, a selection's and a bias's arms read no per-lane mask"
            ),
            Self::KvWrite { lane, why } => write!(f, "lane {lane} refused: {why}"),
            Self::Overreach {
                instance,
                lanes,
                carried,
            } => write!(
                f,
                "instance {instance} is attached over lanes {}..{} and its `embed_indptr` port \
                 describes {carried} lane(s): a program's ports state its own lanes' geometry, \
                 never the next request's",
                lanes.start, lanes.end
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
            Self::OutOfMemory { need, have } => write!(
                f,
                "this load wants {need} bytes of device memory and the device has \
                 {have} free"
            ),
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
            Self::Claimed(refusal) => write!(f, "`guest::claims` refused: {refusal}"),
            Self::Integrity { at, why } => write!(f, "{at}: {why} (a fault in this engine)"),
            Self::TierFull(why) => write!(f, "the disk tier's slot file is full: {why}"),
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
            Self::Region { node, body, why } => write!(
                f,
                "the `fused.region` at node {node} (body {body:016x}) breaks the row skeleton's \
                 contract: {why}. poem-compiler folds only chains the skeleton takes, so the fold \
                 formed one it should have left as its ops"
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

impl From<poem_compiler::Error> for Fault {
    fn from(refusal: poem_compiler::Error) -> Fault {
        Fault::Bake(refusal)
    }
}

impl From<checkpoint::error::Error> for Fault {
    fn from(error: checkpoint::error::Error) -> Fault {
        Fault::Load(error)
    }
}

impl From<poem_exec::Error> for Fault {
    fn from(error: poem_exec::Error) -> Fault {
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

impl From<engine::ports::PortError> for Fault {
    fn from(error: engine::ports::PortError) -> Fault {
        Fault::program("program::ports", error.0)
    }
}

impl From<poem_exec::KernelError> for Fault {
    fn from(error: poem_exec::KernelError) -> Fault {
        Fault::Fire(poem_exec::Error::Kernel(error))
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
