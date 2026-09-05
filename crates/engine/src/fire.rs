//! The fire submission: one forward pass over the batch the runtime assembled.

use serde::{Deserialize, Serialize};

use crate::channel::Ticket;
use crate::error::{Error, Result};
use crate::program::InstanceId;

/// A fire's id, minted by the engine, unique for the life of a load.
pub type FireId = u64;

/// A frame's id, minted by the engine, unique for the life of a load. One
/// frame is one [`submit`](crate::Engine::submit): 1..=k steps, sealed in
/// order, admitted together.
pub type FrameId = u64;

/// Which readable extent of a slot a lane's attention may reach.
///
/// Run-length encoding: alternating masked-out/kept lengths, starting
/// masked-out. `total` may exceed the lane's readable extent (clipped at the
/// causal bound) but must not fall short (refused rather than silently
/// truncated).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mask {
    /// Alternating run lengths, masked-out first.
    pub runs: Vec<u32>,
    /// How many positions the runs cover.
    pub total: u64,
}

impl Mask {
    /// The mask these runs describe.
    #[must_use]
    pub fn new(runs: Vec<u32>, total: u64) -> Mask {
        Mask { runs, total }
    }

    /// How many positions it covers.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.total
    }

    /// True when it covers nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// How many `u32` words a bitmap of this mask occupies.
    #[must_use]
    pub fn words(&self) -> usize {
        usize::try_from(self.total.div_ceil(32)).unwrap_or(usize::MAX)
    }

    /// Expand into a bitmap: bit `i` set iff position `i` is kept. `dst` must
    /// hold at least [`Mask::words`] entries and is zeroed by the caller.
    pub fn expand_into(&self, dst: &mut [u32]) {
        let total = usize::try_from(self.total).unwrap_or(usize::MAX);
        let mut at = 0usize;
        for (index, &run) in self.runs.iter().enumerate() {
            let end = at.saturating_add(run as usize).min(total);
            if index % 2 == 1 {
                for bit in at..end {
                    if let Some(word) = dst.get_mut(bit / 32) {
                        *word |= 1 << (bit % 32);
                    }
                }
            }
            if end == total {
                break;
            }
            at = end;
        }
    }
}

/// How a lane's attention is restricted: over its extent, or per row.
///
/// [`Masking::Extent`] applies one mask to every query row under the causal
/// bound; it cannot express a sliding window (rows keep non-nested ranges),
/// which [`Masking::Rows`] does with one [`Mask`] per token. Both are
/// intersected with `k <= held + row` on expansion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Masking {
    /// One restriction over the lane's readable extent, applied to every row.
    Extent(Mask),
    /// One restriction per query row, parallel to [`Lane::tokens`].
    Rows(Vec<Mask>),
}

impl Masking {
    /// The masks this states, in row order.
    #[must_use]
    pub fn masks(&self) -> &[Mask] {
        match self {
            Masking::Extent(mask) => std::slice::from_ref(mask),
            Masking::Rows(rows) => rows,
        }
    }

    /// The mask query row `row` reads under, or `None` for a row this
    /// masking does not describe.
    #[must_use]
    pub fn of_row(&self, row: usize) -> Option<&Mask> {
        match self {
            Masking::Extent(mask) => Some(mask),
            Masking::Rows(rows) => rows.get(row),
        }
    }

    /// How many query rows this masking states one each for, or `None` for
    /// [`Masking::Extent`].
    #[must_use]
    pub fn stated_rows(&self) -> Option<usize> {
        match self {
            Masking::Extent(_) => None,
            Masking::Rows(rows) => Some(rows.len()),
        }
    }
}

/// What this fire does to a lane's KV. Token count written is
/// `Lane::tokens.len()`, not stated separately here.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDelta {
    /// How many tokens this lane's slot already holds — the first position
    /// this fire writes. Zero on a first prefill.
    pub held: u32,
    /// The KV pages this lane may address, in sequence order. Empty means the
    /// shell owns the page table for this slot; non-empty means the runtime
    /// keeps it (needed for an exported [`KvHandle`](crate::transfer::KvHandle)).
    /// Pool page ids, not the guest's own page space (see
    /// [`KvDelta::translation`]).
    pub pages: Vec<u32>,
    /// The working set's flat table: entry `i` is the pool page backing
    /// working-set-relative index `i`. Empty for every lane whose page
    /// references the runtime has already resolved (every class but
    /// [`GeometryClass::DeviceGeometry`](eta_ir::registry::GeometryClass)).
    /// Minted only by the KV store; an engine may only index it, so empty
    /// beside unresolved device-geometry page references is a refusal, not a
    /// default.
    #[serde(default)]
    pub translation: Vec<u32>,
}

/// Which of a lane's rows the caller reads back.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Readout {
    /// The last row only — what a sampler wants, and the reason a prefill does
    /// not hand back a 0.5 MB logits row per teacher-forced position.
    #[default]
    Last,
    /// These rows of this lane, by index within the lane.
    Rows(Vec<u32>),
    /// Nothing: the lane runs for its cache writes alone.
    None,
}

/// One request inside a fire, as the runtime submits it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lane {
    /// Which pool slot this request's sequence lives in.
    pub slot: u32,
    /// The lane's fact bits, indexed by `Guard::Fact(bit)`; computed by the
    /// model's `Classify::of`. A word the loaded artifact has no class for
    /// is a refusal.
    pub word: u64,
    /// Token ids fed this fire — a prompt on the first fire, one token after,
    /// `1 + drafts` under speculation. Also the lane's row count.
    pub tokens: Vec<u32>,
    /// Each token's position in its sequence. Empty means the natural run
    /// `held .. held + tokens.len()`; non-empty for a speculative re-feed of
    /// rejected positions or an mRoPE lane with non-1-D positions.
    pub positions: Vec<u32>,
    /// What this fire does to the lane's cache.
    pub kv: KvDelta,
    /// An explicit attention mask, replacing the derived causal one; `Some`
    /// makes the lane's `masked` fact true. A [`Masking`] rather than a bare
    /// [`Mask`] so a sliding window is expressible.
    pub mask: Option<Masking>,
    /// Which adapter bank this lane routes to. `None` is the base model.
    pub adapter: Option<u32>,
    /// Run the model's draft head over this lane's rows; must agree with
    /// [`Lane::word`]'s class. Row alignment is the caller's: the head reads
    /// `(hidden at p, token at p+1)`, so row `r` must carry the token one
    /// position past the hidden the trunk leaves at `r`.
    pub drafts: bool,
    /// Keep this lane's attention mass; puts its per-layer log-sum-exp into
    /// [`LaneReadout::scores`]. [`Lane::drafts`]'s twin in validation.
    pub captures_scores: bool,
    /// Every row attends every key of the lane's readable extent, the keys
    /// this fire writes after the row included — a denoiser's reading of a
    /// canvas. `false` is the causal reading. Lifted on the custom-mask arm,
    /// which is why a bidirectional lane must carry a [`Lane::mask`] (an
    /// all-keeping one when it has nothing else to say); served only by an
    /// engine declaring
    /// [`bidirectional_attention`](crate::Capabilities::bidirectional_attention),
    /// refused by name elsewhere rather than silently read causally.
    #[serde(default)]
    pub bidirectional: bool,
    /// A denoiser's self-conditioning taps for this lane's rows
    /// (`RuntimeInput::SelfCondRows/Weights`); `None` for a lane with no
    /// signal, which an engine stages as zero weights when the plan reads
    /// the input at all. Only meaningful beside [`Lane::bidirectional`].
    #[serde(default)]
    pub self_cond: Option<SelfCondInput>,
    /// The recurrent-state verb this lane asks for ([`RsVerb`]). An engine
    /// that does not serve the non-default verbs
    /// ([`Serves::rs_verbs`]) refuses them by name rather than folding.
    #[serde(default)]
    pub rs: RsVerb,
    /// Whether this lane's recurrent slot arrives fresh. Owned by the RS
    /// store; not derivable from `kv.held == 0` (fork/restore/seat reuse can
    /// disagree). [`RsReset::Inferred`] (default) is that old rule.
    #[serde(default)]
    pub rs_reset: RsReset,
    /// What this lane predicts its channels' cursors will be. Empty means no
    /// prediction. Accepted only by an engine declaring
    /// [`device_channel_commit`](crate::Capabilities::device_channel_commit);
    /// others refuse it by name.
    #[serde(default)]
    pub channels: Vec<Ticket>,
    /// Which rows come back.
    pub readout: Readout,
}

impl Lane {
    /// A decode lane: one token, no mask, no adapter, last-row readout.
    #[must_use]
    pub fn decode(slot: u32, word: u64, token: u32, held: u32) -> Lane {
        Lane {
            slot,
            word,
            tokens: vec![token],
            kv: KvDelta {
                held,
                ..KvDelta::default()
            },
            ..Lane::default()
        }
    }

    /// How many token rows this lane contributes.
    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.tokens.len()).unwrap_or(u32::MAX)
    }

    /// Is this lane one the contract describes?
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when the positions do not match the tokens, or
    /// a readout names a row the lane does not have.
    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    /// As [`Lane::validate`], for an engine that states whether it validates
    /// channel tickets on the device
    /// ([`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit)).
    ///
    /// # Errors
    ///
    /// As [`Lane::validate`], plus [`Error::Unsupported`] for a stated ticket
    /// against an engine with no device half to check it.
    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if !self.positions.is_empty() && self.positions.len() != self.tokens.len() {
            return Err(Error::Invalid(format!(
                "lane in slot {} has {} positions for {} tokens",
                self.slot,
                self.positions.len(),
                self.tokens.len()
            )));
        }
        if let Readout::Rows(rows) = &self.readout
            && let Some(&row) = rows.iter().find(|&&r| r >= self.rows())
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} reads row {row} of the {} it has",
                self.slot,
                self.rows()
            )));
        }
        // A per-row mask must be parallel to the rows.
        if let Some(masking) = &self.mask
            && let Some(stated) = masking.stated_rows()
            && stated != self.tokens.len()
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} states {stated} per-row masks for the {} rows it carries",
                self.slot,
                self.rows()
            )));
        }
        if !self.channels.is_empty() && !serves.device_channel_commit {
            return Err(Error::unsupported("engine", F3_CHANNEL_TICKETS));
        }
        if self.bidirectional && self.mask.is_none() {
            return Err(Error::Invalid(format!(
                "lane in slot {} attends bidirectionally and states no mask; the custom-mask \
                 arm is where the causal bound is lifted, so the lane must carry one",
                self.slot
            )));
        }
        if self.bidirectional && !serves.bidirectional {
            return Err(Error::unsupported("engine", BIDIRECTIONAL_WITHOUT_ARM));
        }
        if let Some(sc) = &self.self_cond {
            let cells = self.rows() as usize * sc.taps as usize;
            if sc.taps == 0 || sc.rows.len() != cells || sc.weight_bits.len() != cells {
                return Err(Error::Invalid(format!(
                    "lane in slot {} states a self-conditioning input of {} taps as {} ids and \
                     {} weights for its {} rows",
                    self.slot,
                    sc.taps,
                    sc.rows.len(),
                    sc.weight_bits.len(),
                    self.rows()
                )));
            }
        }
        if !matches!(self.rs, RsVerb::Fold) && !serves.rs_verbs {
            return Err(Error::unsupported("engine", RS_VERBS_WITHOUT_DEVICE_HALF));
        }
        // A host-stated fold boundary must be among this fire's own rows; a
        // device-stated one is clamped by the shell at compose.
        if let RsVerb::Buffer {
            fold: FoldLen::Host(fold),
            replay,
            ..
        } = &self.rs
            && *fold > replay.saturating_add(self.rows())
        {
            return Err(Error::Invalid(if *replay == 0 {
                format!(
                    "lane in slot {} folds {fold} of the {} rows it carries",
                    self.slot,
                    self.rows()
                )
            } else {
                format!(
                    "lane in slot {} folds {fold} of the {} rows it carries plus the {replay} \
                     buffered token(s) it replays ahead of them",
                    self.slot,
                    self.rows()
                )
            }));
        }
        Ok(())
    }
}

/// The verb spelling for a channel prediction this engine cannot check.
const F3_CHANNEL_TICKETS: &str =
    "Lane::channels against an engine without the pull-validate and commit-bump kernels";

/// The verb spelling for the recurrent verbs an engine has no device half for.
const RS_VERBS_WITHOUT_DEVICE_HALF: &str = "Lane::rs beyond RsVerb::Fold: this engine has no device half for it";
const BIDIRECTIONAL_WITHOUT_ARM: &str =
    "Lane::bidirectional: this engine's attention applies its own causal bound and cannot lift it";

/// What an engine states it will actually honour, carried into
/// [`Lane::validate_for`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Serves {
    /// This engine advances its channel rings on the device
    /// ([`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit)).
    pub device_channel_commit: bool,
    /// This engine serves [`RsVerb::Buffer`] and [`RsVerb::FoldBuffered`]
    /// ([`Capabilities::rs_verbs`](crate::Capabilities::rs_verbs)).
    pub rs_verbs: bool,
    /// This engine serves [`Lane::bidirectional`]
    /// ([`Capabilities::bidirectional_attention`](crate::Capabilities::bidirectional_attention)).
    pub bidirectional: bool,
}

impl Serves {
    /// Nothing honoured.
    pub const NONE: Serves = Serves {
        device_channel_commit: false,
        rs_verbs: false,
        bidirectional: false,
    };
}

/// A denoiser lane's self-conditioning taps: per row, `taps` token ids and
/// their weights, row major. The weights ride as their `f32` bit patterns
/// so the lane stays `Eq` (a fire is compared and hashed whole).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfCondInput {
    /// Taps per row; the plan's own count.
    pub taps: u32,
    /// `rows() * taps` token ids.
    pub rows: Vec<u32>,
    /// `rows() * taps` weights, as `f32::to_bits`.
    pub weight_bits: Vec<u32>,
}

impl SelfCondInput {
    /// The taps from ids and weights, row major.
    #[must_use]
    pub fn new(taps: u32, rows: Vec<u32>, weights: &[f32]) -> SelfCondInput {
        SelfCondInput {
            taps,
            rows,
            weight_bits: weights.iter().map(|w| w.to_bits()).collect(),
        }
    }

    /// The weights, as numbers.
    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.weight_bits.iter().map(|&bits| f32::from_bits(bits))
    }
}

/// Whether a lane's recurrent slot begins here. Opening a sequence in a slot
/// another sequence used means zeroing what that one left, since the scan
/// reads the whole bank on its first step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsReset {
    /// No statement: the engine's old rule applies (`kv.held == 0` means a
    /// sequence beginning).
    #[default]
    Inferred,
    /// The RS store classified this slot as fresh: zero its banks first.
    Fresh,
    /// The RS store classified this slot as continuing: leave its banks alone.
    Held,
}

/// What this lane's pass does to its recurrent state. The default is the
/// only shape this tree serves today; the other two are the Mamba-family
/// speculation vocabulary, refused by name rather than silently folded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsVerb {
    /// Fold each token into the recurrent state inside the forward.
    #[default]
    Fold,
    /// Scatter the in-projection activations into `pages`, leaving folded
    /// state untouched: a draft whose rejection is pure host bookkeeping.
    Buffer {
        /// The lane's whole buffer run as physical buffer-page slot ids, in
        /// buffer order: entry `j` holds tokens `[j*page_tokens, (j+1)*page_tokens)`.
        /// A list, not a range: pages are copy-on-write after a fork, so a
        /// run is contiguous only by luck.
        pages: Vec<u32>,
        /// Which buffer token this fire's first row lands at.
        at: u32,
        /// How many of this fire's rows this lane also folds.
        /// `FoldLen::Host(0)` is the pure scatter; otherwise the fire also
        /// lands durable state on row `fold`. Counted in the lane's EXTENDED
        /// layout `[replay | rows]`, so it is bounded by `replay + rows`.
        fold: FoldLen,
        /// **The buffer read path**: how many already-buffered tokens sit
        /// immediately before `at` and must be replayed through the
        /// recurrence AHEAD of this fire's rows, so the rows start from
        /// `folded (+) replay(buffer)` rather than from the folded state
        /// alone. The tokens live at buffer positions `[at - replay, at)`.
        /// Zero is the ordinary scatter onto an empty buffer. A speculative
        /// decoder's every round but the first has one: the accepted prefix
        /// of the last window survives in the buffer unfolded, and the next
        /// window's fire folds it (`fold == replay`) while buffering its own
        /// rows. Engines that serve no read path refuse a non-zero value by
        /// name.
        #[serde(default)]
        replay: u32,
    },
    /// **The device-resident speculative round.** The lane's buffer is two
    /// page runs used in alternation: `read` holds the previous window at
    /// buffer token 0, `write` receives this fire's rows at buffer token 0.
    /// The recurrence replays the first `fold` tokens of `read` AHEAD of the
    /// rows and persists its state exactly after them — the previous
    /// window's accepted prefix, which is always its first `fold` rows — so
    /// no token count, offset or discard ever reaches the host: `fold` is
    /// the accepted count the verifying epilogue computed and put on the
    /// `rs_fold_len` channel. The runtime alternates the two runs per fire.
    Window {
        /// The previous window's pages, buffer order; may be empty on the
        /// first round (nothing to replay).
        read: Vec<u32>,
        /// This window's pages, buffer order.
        write: Vec<u32>,
        /// How many of `read`'s tokens are replayed and folded.
        fold: FoldLen,
    },
    /// Replay the buffer through conv+recurrence, truncated at the accepted
    /// boundary: the batch fold, skipping the in-projection GEMM.
    FoldBuffered {
        /// The lane's buffer run, addressed exactly as [`RsVerb::Buffer`]
        /// wrote it.
        pages: Vec<u32>,
        /// Buffer-token offset the replay starts at — [`RsVerb::Buffer`]'s
        /// `at`. A fold can only release whole covered pages, so survivors
        /// may sit offset inside their first page; this is that offset.
        at: u32,
        /// The host's upper bound on the accepted length, sizing the launch.
        /// The device clamps to it.
        bound: u32,
        /// The accepted length itself.
        len: FoldLen,
    },
}

/// How long a fold is, and who knows it. The accepted count is device data
/// computed by the verifier and must not round-trip through the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldLen {
    /// The host knows it: a fixed count it computed itself.
    Host(u32),
    /// The device knows it: read this descriptor port at compose time.
    Device(eta_ir::registry::Port),
}

/// Where a guest program runs relative to the immutable graph: before or
/// after, never mid-graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Boundary {
    /// Before the graph: token preparation, channel reads, state.
    Prologue,
    /// After the graph: sampling, decode logic, channel commit.
    Epilogue,
}

/// One guest-program instance attached to this fire. One attachment per
/// instance per fire: `Stage::Prologue`/`Stage::Epilogue` are one pass with
/// one commit. A program reading [`IntrinsicId::Logits`](eta_ir::op::IntrinsicId)
/// must be [`Boundary::Epilogue`]. Naming one instance twice is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    /// Which lane of this submission it runs for.
    pub lane: u32,
    /// Which bound instance.
    pub instance: InstanceId,
    /// Which end of the graph.
    pub at: Boundary,
}

/// One lane's media spans, as the submission carries them. A parallel slice
/// keyed by lane, like [`Attachment`]: most lanes have no media.
///
/// `patches` is `f32`; conversion to the plan's element type happens at the
/// engine's marshal.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepMedia {
    /// Which lane of this step the spans belong to; rebased by the batcher
    /// like [`Attachment::lane`].
    pub lane: u32,
    /// Payload rows per span, in submission order; length is the span count,
    /// sum is the payload row count.
    pub rows: Vec<u32>,
    /// Payload rows concatenated over this lane's spans:
    /// `rows.iter().sum()` rows of the plan's declared patch width.
    pub patches: Vec<f32>,
    /// Where this lane's tower output lands in the token rectangle — one
    /// entry per payload row, as an offset into this lane's token rows.
    /// Lane-relative; `-1` names no row and is not rebased.
    pub routes: Vec<i32>,
    /// The tower's rotation stream: three `i32` per payload row, each row's
    /// `(t, h, w)` in its span's grid. Not rebased (a grid coordinate is a
    /// span property, same in every fire it lands in).
    pub positions: Vec<i32>,
    /// Which rows of the learned position table each row gathers: `taps`
    /// entries per payload row. Empty on the native grid (no resampling).
    pub embed_rows: Vec<i32>,
    /// Beside [`embed_rows`](StepMedia::embed_rows), same length.
    pub embed_weights: Vec<f32>,
    /// The trunk's rotation stream for this lane: three `i32` per token row.
    /// Empty means scalar `(p, p, p)` (1-D RoPE); M-RoPE owes one triple per row.
    pub token_positions: Vec<i32>,
}

impl StepMedia {
    /// How many payload rows this lane's spans contribute in total.
    #[must_use]
    pub fn payload_rows(&self) -> u32 {
        self.rows.iter().copied().fold(0u32, u32::saturating_add)
    }

    /// Checks only what a length function of `rows` alone can settle; payload
    /// width and tap count are the plan's own numbers, refused at the shell
    /// instead (`Fault::PatchPayload`).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] with the first thing that is wrong.
    pub fn validate(&self, lane_rows: u32) -> Result<()> {
        let rows = self.payload_rows();
        if self.rows.is_empty() {
            return Err(Error::Invalid(format!(
                "lane {} carries a media row naming no spans; a lane with no span \
                 constructs no media row at all",
                self.lane
            )));
        }
        if rows == 0 {
            return Err(Error::Invalid(format!(
                "lane {}'s spans occupy no payload rows, and a span the tower reads \
                 nothing of has nothing to scatter",
                self.lane
            )));
        }
        for (what, have, owed) in [
            ("routes", self.routes.len(), rows as usize),
            ("grid positions", self.positions.len(), 3 * rows as usize),
        ] {
            if have != owed {
                return Err(Error::Invalid(format!(
                    "lane {}'s media carries {have} {what} for {rows} payload rows, \
                     and {owed} are owed",
                    self.lane
                )));
            }
        }
        if self.embed_rows.len() != self.embed_weights.len() {
            return Err(Error::Invalid(format!(
                "lane {}'s media carries {} position-table rows and {} weights; the \
                 two streams are read together and are the same length or both empty",
                self.lane,
                self.embed_rows.len(),
                self.embed_weights.len()
            )));
        }
        if !self.token_positions.is_empty()
            && self.token_positions.len() != 3 * lane_rows as usize
        {
            return Err(Error::Invalid(format!(
                "lane {}'s trunk rotation stream carries {} entries for {lane_rows} \
                 token rows; it is empty (scalar `(p, p, p)`) or three per row",
                self.lane,
                self.token_positions.len()
            )));
        }
        Ok(())
    }
}

/// One forward pass over the assembled batch — one step of a frame (1..=k
/// steps, sealed in order).
///
/// No `Eq`: a payload row is `f32`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// The requests in this fire, in submission order. Answers come back in
    /// the same order.
    pub lanes: Vec<Lane>,
    /// The guest programs attached at this fire's boundaries.
    pub attachments: Vec<Attachment>,
    /// The media spans this fire's lanes submitted, keyed by lane; empty for
    /// every text-only fire.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<StepMedia>,
}

impl Step {
    /// How many token rows the whole fire carries.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.lanes.iter().map(Lane::rows).sum()
    }

    /// Is this a submission the contract describes? Checks per-lane
    /// arithmetic plus batch-wide invariants: no slot twice, every
    /// attachment names an existing lane.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] with the first thing that is wrong.
    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    /// As [`Step::validate`], carrying each engine's own answer about
    /// channel tickets down to [`Lane::validate_for`].
    ///
    /// # Errors
    ///
    /// As [`Step::validate`].
    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if self.lanes.is_empty() {
            return Err(Error::Invalid("a fire carries no lanes".into()));
        }
        for (index, lane) in self.lanes.iter().enumerate() {
            lane.validate_for(serves)?;
            if self.lanes[..index].iter().any(|l| l.slot == lane.slot) {
                return Err(Error::Invalid(format!(
                    "slot {} appears twice in one fire, at lane {index}",
                    lane.slot
                )));
            }
        }
        let lanes = u32::try_from(self.lanes.len()).unwrap_or(u32::MAX);
        for (index, attachment) in self.attachments.iter().enumerate() {
            if attachment.lane >= lanes {
                return Err(Error::Invalid(format!(
                    "attachment names lane {} of the {lanes} this fire has",
                    attachment.lane
                )));
            }
            // One pass per instance per fire: see [`Attachment`].
            if self.attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attachment.instance)
            {
                return Err(Error::Invalid(format!(
                    "instance {} is attached twice to one fire, at attachment {index}; a \
                     program's stages are one pass with one commit",
                    attachment.instance
                )));
            }
        }
        // Same batch-wide checks as an attachment, plus StepMedia::validate.
        for (index, media) in self.media.iter().enumerate() {
            let Some(lane) = self.lanes.get(media.lane as usize) else {
                return Err(Error::Invalid(format!(
                    "media row {index} names lane {} of the {lanes} this fire has",
                    media.lane
                )));
            };
            if self.media[..index].iter().any(|earlier| earlier.lane == media.lane) {
                return Err(Error::Invalid(format!(
                    "lane {} carries two media rows, at media row {index}; a lane's \
                     spans are one concatenation with one payload order",
                    media.lane
                )));
            }
            media.validate(lane.rows())?;
        }
        Ok(())
    }
}

/// The unit of work the contract admits: 1..=k steps, sealed in order,
/// validated and committed together as one frame rather than a partial one
/// the caller can neither retry nor undo.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameSubmission {
    /// The steps, in the order the device runs them. At least one.
    pub steps: Vec<Step>,
}

impl FrameSubmission {
    /// The degenerate one-step frame.
    #[must_use]
    pub fn of(step: Step) -> FrameSubmission {
        FrameSubmission { steps: vec![step] }
    }

    /// How many token rows the whole frame carries, across its steps.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.steps.iter().map(Step::rows).sum()
    }

    /// Is this a frame the contract describes? Every step is checked before
    /// any is admitted.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] with the first thing that is wrong, and
    /// [`Error::Unsupported`] for a step naming a shape this tree does not
    /// serve yet.
    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    /// As [`FrameSubmission::validate`], carrying the admitting engine's own
    /// [`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit)
    /// down to [`Lane::validate_for`].
    ///
    /// # Errors
    ///
    /// As [`FrameSubmission::validate`].
    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if self.steps.is_empty() {
            return Err(Error::Invalid("a frame carries no steps".into()));
        }
        for step in &self.steps {
            step.validate_for(serves)?;
        }
        Ok(())
    }
}

/// The receipt for one admitted frame, one entry per step. A synchronous
/// shell fills every step's readouts before `submit` returns; an
/// asynchronous one answers with empty readouts and correlates completion
/// on [`FrameTicket::id`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameTicket {
    /// This frame's id, unique for the life of the load.
    pub id: FrameId,
    /// One receipt per submitted step, in submission order.
    pub steps: Vec<FireTicket>,
}

/// One attention layer's captured mass, for one lane. The log-sum-exp (the
/// normalizer per-key scores are a ratio against), not a full score matrix —
/// a paged attention kernel never materializes that.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LayerScores {
    /// Which transformer layer this column came from, as the plan's `Seam`
    /// row stamps it.
    pub layer: u32,
    /// How many of the lane's rows came back.
    pub rows: u32,
    /// How many query heads each row holds.
    pub heads: u32,
    /// The mass, row-major, `rows * heads` of them.
    pub lse: Vec<f32>,
}

/// What one lane read back.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LaneReadout {
    /// How many rows came back.
    pub rows: u32,
    /// How wide each row is — the vocabulary, for logits.
    pub width: u32,
    /// The values, row-major, `rows * width` of them.
    pub values: Vec<f32>,
    /// This lane's captured attention mass, one entry per attention layer the
    /// model text exports at `model_dsl::seam::SCORES`, in layer order. Empty
    /// unless the lane set [`Lane::captures_scores`].
    #[serde(default)]
    pub scores: Vec<LayerScores>,
}

/// The receipt for one accepted fire. A synchronous shell fills
/// [`FireTicket::readouts`] before returning; an asynchronous one answers
/// with the id and empty readouts, correlating completion on
/// [`FireTicket::id`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FireTicket {
    /// This fire's id, unique for the life of the load.
    pub id: FireId,
    /// One entry per submitted lane, in submission order. Empty from an engine
    /// that answers before the device is done.
    pub readouts: Vec<LaneReadout>,
}

/// The `encode` verb's argument: non-text modalities in, embedding rows out.
/// A batch of independent blobs with an anchor row each.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaEncode {
    /// Per image: `(temporal, height, width)` patch counts, three entries each.
    pub image_grids: Vec<u32>,
    /// Pixel bytes for every image, `f32`-aligned.
    pub image_pixels: Vec<u8>,
    /// Byte offsets splitting `image_pixels` per image; one more than there
    /// are images.
    pub image_pixel_indptr: Vec<u32>,
    /// Per patch: its `(y, x)` position, two entries each.
    pub image_patch_positions: Vec<u32>,
    /// Which output row each image's embeddings anchor at.
    pub image_anchor_rows: Vec<u32>,
    /// Feature bytes for every audio clip, `f32`-aligned.
    pub audio_features: Vec<u8>,
    /// Byte offsets splitting `audio_features` per clip.
    pub audio_feature_indptr: Vec<u32>,
    /// Which output row each clip's embeddings anchor at.
    pub audio_anchor_rows: Vec<u32>,
    /// The embedding rows, `bf16`, filled by the engine.
    pub output_rows: Vec<u8>,
    /// Byte offsets splitting `output_rows` per image then per clip.
    pub output_row_indptr: Vec<u32>,
}

impl MediaEncode {
    /// Is this an encode the contract describes?
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a payload with no anchor, an anchor with no
    /// payload, or a partition that does not cover its bytes.
    pub fn validate(&self) -> Result<()> {
        const F32: usize = size_of::<f32>();
        const U16: usize = size_of::<u16>();
        let bad = |why: String| Err(Error::Invalid(why));

        let images = self.image_anchor_rows.len();
        let clips = self.audio_anchor_rows.len();
        if images + clips == 0 {
            return bad("an encode carries no image and no audio anchor".into());
        }
        if self.output_row_indptr.len() != images + clips + 1 {
            return bad(format!(
                "output_row_indptr has {} entries for {images} images and {clips} clips",
                self.output_row_indptr.len()
            ));
        }
        if self.output_rows.is_empty() || !self.output_rows.len().is_multiple_of(U16) {
            return bad(format!(
                "output_rows is {} bytes, which is empty or not a whole number of u16",
                self.output_rows.len()
            ));
        }

        if images == 0 {
            if !self.image_grids.is_empty()
                || !self.image_pixels.is_empty()
                || !self.image_pixel_indptr.is_empty()
                || !self.image_patch_positions.is_empty()
            {
                return bad("an image payload arrived with no image anchor to attach it to".into());
            }
        } else {
            if self.image_grids.len() != images.saturating_mul(3) {
                return bad(format!(
                    "image_grids has {} entries for {images} images",
                    self.image_grids.len()
                ));
            }
            if self.image_pixel_indptr.len() != images + 1 {
                return bad(format!(
                    "image_pixel_indptr has {} entries for {images} images",
                    self.image_pixel_indptr.len()
                ));
            }
            if self.image_pixels.is_empty() || !self.image_pixels.len().is_multiple_of(F32) {
                return bad(format!(
                    "image_pixels is {} bytes, which is empty or not a whole number of f32",
                    self.image_pixels.len()
                ));
            }
            if self.image_patch_positions.is_empty()
                || !self.image_patch_positions.len().is_multiple_of(2)
            {
                return bad(format!(
                    "image_patch_positions has {} entries, which is empty or not a whole number \
                     of pairs",
                    self.image_patch_positions.len()
                ));
            }
            partition(
                &self.image_pixel_indptr,
                "image_pixel_indptr",
                self.image_pixels.len(),
                F32,
                false,
            )?;
        }

        if clips == 0 {
            if !self.audio_features.is_empty() || !self.audio_feature_indptr.is_empty() {
                return bad("an audio payload arrived with no audio anchor to attach it to".into());
            }
        } else {
            if self.audio_feature_indptr.len() != clips + 1 {
                return bad(format!(
                    "audio_feature_indptr has {} entries for {clips} clips",
                    self.audio_feature_indptr.len()
                ));
            }
            if self.audio_features.is_empty() || !self.audio_features.len().is_multiple_of(F32) {
                return bad(format!(
                    "audio_features is {} bytes, which is empty or not a whole number of f32",
                    self.audio_features.len()
                ));
            }
            partition(
                &self.audio_feature_indptr,
                "audio_feature_indptr",
                self.audio_features.len(),
                F32,
                true,
            )?;
        }
        Ok(())
    }
}

/// Does `indptr` partition `bytes` into `align`-aligned, ordered segments?
fn partition(
    indptr: &[u32],
    name: &str,
    bytes: usize,
    align: usize,
    strict: bool,
) -> Result<()> {
    if indptr.first().copied() != Some(0) {
        return Err(Error::Invalid(format!(
            "{name} starts at {:?}, not 0",
            indptr.first()
        )));
    }
    if indptr.last().copied() != u32::try_from(bytes).ok() {
        return Err(Error::Invalid(format!(
            "{name} ends at {:?}, not the {bytes} bytes it partitions",
            indptr.last()
        )));
    }
    for w in indptr.windows(2) {
        let ordered = if strict { w[0] < w[1] } else { w[0] <= w[1] };
        if !ordered {
            return Err(Error::Invalid(format!(
                "{name} segment {}..{} is empty or inverted",
                w[0], w[1]
            )));
        }
        if !(w[0] as usize).is_multiple_of(align) || !(w[1] as usize).is_multiple_of(align) {
            return Err(Error::Invalid(format!(
                "{name} segment {}..{} is not {align}-byte aligned",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}
