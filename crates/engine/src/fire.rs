//! The fire submission — one forward pass over the batch the runtime assembled.
//!
//! ```text
//! fire (R=5):        lane0 prefill(7 rows)  lane1 prefill(3)  lane2..4 decode(1 each)
//! rows (seriated):   [··············· 10 ···············|········ 3 ········]
//! ```
//!
//! # `Lane::word` is the one genuinely new field
//!
//! Everything else in this module is a rename of something the old
//! `LaunchPlan` carried. [`Lane::word`] is not: it is the per-lane fact bits
//! `Classify::of(&Request)` computed runtime-side, and it is what
//! `model_exec::fire::compose` turns into a class, and therefore into the row
//! WINDOW every guarded node of the artifact runs over (design §0,
//! decision 18).
//!
//! It is per-LANE and this matters. The contract it replaces had one fact word
//! per FIRE — `FireBindings::facts` — which forced a mixed batch to be split
//! into a decode fire and a prefill fire, or to be run through a kernel
//! general enough for both. Window-split is the mechanism that makes neither
//! necessary, and a per-fire word is exactly the collapse that makes
//! window-split unexpressible.
//!
//! # Why the submission is this small
//!
//! The `LaunchPlan` this replaces had **62 fields**, most of them parallel
//! `Vec<u32>` CSR arms — `qo_indptr`, `kv_page_indptr`, `rs_translation_indptr`,
//! `embed_block_indptr`, `image_mrope_indptr`, … — plus a 420-line
//! `StepSubmission::validate` whose job was to check that they were all the
//! same length as each other. That is a serialized data structure, not a
//! submission: the runtime flattened its per-request state into eleven CSRs,
//! and the engine's first act was to walk them back into per-request form.
//!
//! Here a lane is a lane. The CSRs are the SHELL's — it builds them in
//! `compose`, from a `Vec<Lane>` — and a whole class of "these two arms
//! disagree" failures cannot be submitted. What is left to validate is
//! arithmetic about one lane at a time.

use serde::{Deserialize, Serialize};

use crate::channel::Ticket;
use crate::error::{Error, Result};
use crate::program::InstanceId;

/// A fire's id, minted by the engine, unique for the life of a load.
pub type FireId = u64;

/// A frame's id, minted by the engine, unique for the life of a load.
///
/// One frame is one [`submit`](crate::Engine::submit): 1..=k steps, sealed in
/// order, admitted together (alto design §1 article 4).
pub type FrameId = u64;

/// Which readable extent of a slot a lane's attention may reach.
///
/// A run-length encoding: alternating lengths of masked-out and kept
/// positions, starting with masked-out. `total` is how many positions the runs
/// describe, so a truncated run list is a detectable submission rather than a
/// silently short mask.
///
/// This is `EncodedMask`, kept: an explicit mask is a real axis (the `masked`
/// fact) and the run form is genuinely how a mask arrives — a prefix drop, a
/// sliding window, a set of retained blocks are all a handful of runs over
/// thousands of positions.
///
/// # `total` MAY EXCEED the lane's readable extent, and may not fall short
///
/// A caller states its mask over the width it KNOWS, and for a guest that is
/// the pool it reserved — 48 key positions for a three-page pool of sixteen,
/// while the lane holds 23 tokens — because the reservation's width does not
/// move as the sequence grows a token per fire. An engine expands the runs
/// into a `rows x extent` rectangle and CLIPS the surplus: a position past
/// `KvDelta::held + tokens.len()` is one this fire has not written, so the
/// causal bound drops it for every query row whatever its bit says, and
/// dropping it is therefore not a choice between readings.
///
/// The other direction is not symmetric and is not accepted. A mask whose
/// `total` is SHORT of the extent expands with its tail bits zero, and zero
/// is MASKED-OUT — an attention silently truncated to the stated prefix. That
/// is what `total` being stated at all is for: a truncated run list is a
/// detectable submission, and the engines refuse it by name
/// (`engine_cuda::Fault::Mask`) rather than pad it.
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

    /// Expand into a bitmap: bit `i` set iff position `i` is KEPT.
    ///
    /// Writes into `dst`, which must hold at least [`Mask::words`] entries.
    /// Anything past the runs stays as it was, which is why the buffer is
    /// zeroed by the caller rather than here — a shell expands many lanes'
    /// masks into one slab.
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

/// **How a lane's attention is restricted: over its EXTENT, or per ROW.**
///
/// [`Mask`] is one run-length restriction of a lane's readable extent, and
/// for most masks that is the whole truth: a prefix drop, a retained-block
/// set, an attention sink are each one set of keys that every query row of
/// the lane reads under, with the causal bound doing the rest. That mask is
/// [`Masking::Extent`], it is what every caller before this type wrote, and
/// an engine re-applies the causal bound to it per row.
///
/// **WHAT THAT SHAPE CANNOT SAY IS A WINDOW.** A sliding-window prefill asks
/// row `i` to keep `[i - w, i]` and row `i + 1` to keep `[i + 1 - w, i + 1]`:
/// two restrictions that are not nested, so no single mask under the causal
/// bound is either of them. The lowering that had only [`Masking::Extent`]
/// to write into refused such a fire by name rather than pick one row — the
/// old one silently picked row ZERO, which is every later row truncated to
/// the first one's causal bound — and [`Masking::Rows`] is the form that
/// refusal was waiting for.
///
/// **`Rows` IS PARALLEL TO THE LANE'S QUERY ROWS**, one [`Mask`] per entry of
/// [`Lane::tokens`], each over that row's own readable extent — which is the
/// lane's post-append extent for every row, the same number
/// [`Masking::Extent`]'s single mask covers, because the causal bound and not
/// the mask is what keeps row `i` off the keys row `i + 1` writes. A count
/// that is not the lane's row count is refused by [`Lane::validate`]: it is
/// the one thing about a per-row mask a lane can check about itself.
///
/// **AN ENGINE MAY NOT WIDEN CAUSALITY WITH EITHER FORM.** Both are
/// intersected with `k <= held + row` on expansion. A mask states which of
/// the readable positions a row may reach, never that a row may reach a
/// position the fire has not written yet.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Masking {
    /// ONE restriction over the lane's readable extent, re-applied to every
    /// query row under the causal bound. Every mask written before per-row
    /// masks existed is this one.
    Extent(Mask),
    /// ONE restriction PER QUERY ROW, parallel to [`Lane::tokens`]: entry `i`
    /// is row `i`'s mask, over the lane's readable extent.
    Rows(Vec<Mask>),
}

impl Masking {
    /// The masks this states, in row order — one for [`Masking::Extent`].
    ///
    /// The one-element slice is not a convenience: it is what lets a shell's
    /// extent check ("does every stated mask cover this lane's extent?") be
    /// one loop over both forms rather than two spellings of one rule.
    #[must_use]
    pub fn masks(&self) -> &[Mask] {
        match self {
            Masking::Extent(mask) => std::slice::from_ref(mask),
            Masking::Rows(rows) => rows,
        }
    }

    /// The mask query row `row` reads under, or `None` for a row this
    /// masking does not describe.
    ///
    /// [`Masking::Extent`] describes EVERY row — that is what "over the
    /// extent" means — so it answers for any index; [`Masking::Rows`]
    /// answers for the rows it carries and for no others.
    #[must_use]
    pub fn of_row(&self, row: usize) -> Option<&Mask> {
        match self {
            Masking::Extent(mask) => Some(mask),
            Masking::Rows(rows) => rows.get(row),
        }
    }

    /// How many query rows this masking states one each for, or `None` for
    /// [`Masking::Extent`], which states one for all of them.
    #[must_use]
    pub fn stated_rows(&self) -> Option<usize> {
        match self {
            Masking::Extent(_) => None,
            Masking::Rows(rows) => Some(rows.len()),
        }
    }
}

/// What this fire does to a lane's KV.
///
/// `held` is the state the shell already has and `pages` is the addressing it
/// may use; the number of tokens WRITTEN is `Lane::tokens.len()`, never stated
/// separately, because a submission where those two numbers disagreed had one
/// of them wrong and no way to tell which.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDelta {
    /// How many tokens this lane's slot already holds — the first position
    /// this fire writes. Zero on a first prefill.
    pub held: u32,
    /// The KV pages this lane may address, in sequence order.
    ///
    /// Empty means the SHELL owns the page table for this slot: it allocated
    /// the pages at `open` and grows them itself. A non-empty list is the
    /// runtime keeping the page table, which is what a load with an exported
    /// [`KvHandle`](crate::transfer::KvHandle) and a remote peer writing into
    /// it needs.
    ///
    /// **THESE ARE POOL PAGE IDS AND NOTHING ELSE.** An engine pushes each
    /// entry straight into its page CSR, so the space is the POOL's — the ids
    /// its own allocator minted. A guest states its pages in a different space
    /// (see [`KvDelta::translation`]) and the runtime translates before it
    /// submits.
    pub pages: Vec<u32>,
    /// **THE WORKING SET'S FLAT TABLE: entry `i` is the pool page backing that
    /// working set's relative index `i`.** Empty for every lane whose page
    /// references the runtime has already resolved, which is every lane of
    /// every class but one.
    ///
    /// # Two spaces, and which side crosses between them
    ///
    /// A guest never holds a pool page id. `kv-working-set`'s whole surface is
    /// WORKING-SET-RELATIVE indexes — `reserve` hands back `0 .. n`, a fork is
    /// O(1) precisely because a relative index survives the copy-on-write that
    /// moves the physical page under it — and the runtime translates through
    /// this table at the point where a page reference stops being the guest's
    /// and becomes an address.
    ///
    /// For every host-resolved geometry that point is inside the runtime:
    /// `pipeline::fire::map_lane_pages` rewrites the folded `Pages` port into
    /// [`KvDelta::pages`] and an engine sees only pool ids.
    ///
    /// [`GeometryClass::DeviceGeometry`](eta_ir::registry::GeometryClass)
    /// is the one class where that point cannot be inside the runtime: the
    /// lane's page ids and its write descriptor are computed by the guest's
    /// own epilogue and live in a channel cell the host never reads. So the
    /// runtime ships the TABLE instead of the result, and the engine applies
    /// it to the `pages`, `page_indptr` and `w_slot` values it resolves off
    /// those cells.
    ///
    /// **THE MAPPING STILL HAS ONE OWNER** (article 8). The KV store mints it
    /// and nothing else may compute one; this field is that table quoted, and
    /// an engine may only index it. An index the table does not cover is a
    /// refusal by name — the alternative is addressing a pool page belonging
    /// to somebody else, which is exactly what the two-spaces confusion costs.
    ///
    /// Empty beside device-resolved page references is therefore a refusal and
    /// not a default: "translate by identity" is the bug, spelled.
    ///
    /// This arm existed once and was deleted in alto E, on the ground that its
    /// only implementation was a refusal — it then meant "rewritten page ids
    /// for a fork that moved this lane's pages", which needed a page mover
    /// neither shell had. It comes back with the meaning
    /// `pipeline::fire::kv`'s `build_translation` was already written for and
    /// documented for ("ships with the launch so the engine can map
    /// channel-resolved `Pages`/`WSlot` references"), against a consumer that
    /// exists.
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
    /// **The lane's fact bits.** `Guard::Fact(bit)` indexes them; the model's
    /// own `Classify::of` computed them (decision 18). A word this load's
    /// artifact has no class for is a refusal, not a default: it means the
    /// runtime and the shell disagree about what is loaded.
    pub word: u64,
    /// The token ids this fire feeds the lane — a prompt on the first fire,
    /// one token on every fire after, `1 + drafts` under speculation.
    ///
    /// **The lane's row count is this length.** There is no separate `rows`
    /// field, for the same reason there is no separate KV write count.
    pub tokens: Vec<u32>,
    /// Each token's position in its sequence.
    ///
    /// Empty means the natural run `held .. held + tokens.len()`, which is
    /// every case but the ones that are the point of stating it: a
    /// speculative fire re-feeding rejected positions, and an mRoPE lane whose
    /// positions are not one-dimensional.
    pub positions: Vec<u32>,
    /// What this fire does to the lane's cache.
    pub kv: KvDelta,
    /// An explicit attention mask, replacing the derived causal one. `Some`
    /// is what makes the lane's `masked` fact true.
    ///
    /// **THE PAYLOAD IS A [`Masking`], NOT A [`Mask`], AND THAT IS THE
    /// SLIDING WINDOW BEING EXPRESSIBLE.** A restriction of the lane's whole
    /// readable extent is [`Masking::Extent`] and is what this field carried
    /// when it was a bare `Mask`; a mask whose ROWS differ — the windowed
    /// prefill, where row `i` keeps `[i - w, i]` — is [`Masking::Rows`], one
    /// mask per entry of [`Lane::tokens`]. `Masking` argues both.
    pub mask: Option<Masking>,
    /// Which adapter bank this lane routes to (design §8). `None` is the base
    /// model.
    pub adapter: Option<u32>,
    /// **Run the model's draft head over this lane's rows** (design §8's MTP
    /// row, palo C3). `true` is what makes the lane's `drafts` fact true.
    ///
    /// **A BARE BOOLEAN, AND THAT IS THE AXIS BEING HONEST ABOUT ITSELF.**
    /// [`Lane::mask`] and [`Lane::adapter`] are payloads whose PRESENCE states
    /// the fact, because each axis needs something from the submission that
    /// only the caller has — a mask's bits, a bank's row. This axis needs
    /// nothing: the draft head reads the lane's own hidden and the lane's own
    /// tokens, over the lane's own rows, and there is no third thing to carry.
    /// So the intent is the whole of the field.
    ///
    /// **IT IS STILL A SECOND STATEMENT AND IT STILL HAS TO AGREE WITH
    /// [`Lane::word`].** The word decides the CLASS, and the class decides
    /// whether the head's arm covers this lane's rows. A lane whose word says
    /// `drafts` and whose submission does not is a fire whose caller will be
    /// handed a draft column it never asked for and will not read; a lane that
    /// asks and whose word does not is a lane that gets no draft and is not
    /// told. Both are refused by name before anything launches
    /// (`engine_cuda::Fault::DraftWord`), which is [`Lane::mask`]'s rule and
    /// [`Lane::adapter`]'s rule for the third time.
    ///
    /// **THE ROW ALIGNMENT IS THE CALLER'S** (`model::qwen_3`'s own note): the
    /// head was trained on `(hidden at p, token at p+1)` and is fed `(x, tok)`
    /// at one row, so a drafting lane's row `r` must carry the token one
    /// position past the hidden the trunk leaves at `r`. The contract states
    /// the requirement; no engine can check it.
    pub drafts: bool,
    /// **Keep this lane's attention mass** (design §9's score-capture
    /// archetype, palo C4). `true` is what makes the lane's `captures_scores`
    /// fact true, and what puts its per-layer log-sum-exp into
    /// [`LaneReadout::scores`].
    ///
    /// [`Lane::drafts`]'s twin in every respect, including the refusal
    /// (`engine_cuda::Fault::ScoreWord`) and including being a bare boolean:
    /// what the capture arm needs is the bit, and everything else about the
    /// observation — which layers, how wide, how many rows — is the model
    /// text's to declare and the artifact's to carry.
    pub captures_scores: bool,
    /// **The recurrent-state verb this lane asks for** (alto design §2/§6).
    ///
    /// The typed verb HEAD never gave a seat: RS was a fold-per-token in the
    /// forward and there was nothing in the contract that could say otherwise,
    /// so speculation over a Mamba family was unexpressible rather than
    /// refused. [`RsVerb`] is that vocabulary.
    ///
    /// **AN ENGINE THAT SERVES THE OTHER TWO SAYS SO** ([`Serves::rs_verbs`],
    /// from [`Capabilities::rs_verbs`](crate::Capabilities::rs_verbs)); every
    /// other refuses them by name at [`Lane::validate`] rather than quietly
    /// folding — a lane that asked for a buffered scatter and got a
    /// destructive fold would have its speculation corrupt the state it was
    /// speculating over.
    #[serde(default)]
    pub rs: RsVerb,
    /// **Whether this lane's recurrent slot arrives fresh** (alto survey §9's
    /// gap list, wave F3).
    ///
    /// The fact belongs to the RS store and to nothing else. Until F3 the
    /// shells derived it from the KV side — a lane stating `kv.held == 0` was
    /// taken to be a sequence beginning, so its recurrent bank was zeroed —
    /// which is a coincidence and not an identity: a runtime that forks a
    /// sequence, restores a prefix or reuses a seat can hand a slot that must
    /// be zeroed while its KV count is non-zero, and can hand one whose KV was
    /// trimmed to nothing while its recurrence must continue.
    ///
    /// [`RsReset::Inferred`] is the default and IS the old rule, restated
    /// where it can be seen: a caller that says nothing gets exactly the
    /// behaviour it had.
    #[serde(default)]
    pub rs_reset: RsReset,
    /// **What this lane predicts its channels' cursors will be** (alto design
    /// §1 article 3). Empty is "the host makes no prediction", which is every
    /// lane in this tree today.
    ///
    /// Accepted by an engine that declares
    /// [`device_channel_commit`](crate::Capabilities::device_channel_commit)
    /// — the pull-validate and commit-bump kernels landed for CUDA in wave
    /// F2a — and refused by name by every other, because a stated-and-ignored
    /// prediction is worse than a refusal (see [`Ticket`]).
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
    /// channel tickets on the device.
    ///
    /// **THE ONE THING A LANE CANNOT CHECK ABOUT ITSELF.** Every other clause
    /// below is a fact about the lane's own shape; whether a stated prediction
    /// will be honoured is a fact about the ENGINE, and
    /// [`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit)
    /// is where an engine says so. An engine that validates them passes
    /// `true`; one that would ignore them passes `false` and the ticket is
    /// refused by name rather than dropped.
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
        // A PER-ROW MASK IS PARALLEL TO THE ROWS OR IT IS NOT A PER-ROW MASK.
        // `Masking::Rows` states one restriction per query row, so a count
        // that is not the lane's row count leaves some row either undescribed
        // (an engine would have to invent one, and inventing it is exactly
        // the silent row-zero substitution this form exists to end) or
        // describes a row the lane does not carry. It is the one thing about
        // a masking a LANE can check about itself: the extent each row covers
        // is `held + rows` and the shell owns that arithmetic, but the count
        // is right here.
        if let Some(masking) = &self.mask
            && let Some(stated) = masking.stated_rows()
            && stated != self.tokens.len()
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} states {stated} per-row masks for the {} rows it                  carries",
                self.slot,
                self.rows()
            )));
        }
        // ── THE TWO F1 SHAPES WITH NO F1 MECHANISM, REFUSED BY NAME.
        //    Both are contract nouns this wave lands so that F2/F3 have
        //    somewhere to put their work; neither has a device half yet, and
        //    an accepted-then-ignored field is the one failure mode a typed
        //    contract exists to prevent (design §1 article 3's own argument
        //    about predictions, and §6's about the fold).
        if !self.channels.is_empty() && !serves.device_channel_commit {
            return Err(Error::unsupported("engine", F3_CHANNEL_TICKETS));
        }
        if !matches!(self.rs, RsVerb::Fold) && !serves.rs_verbs {
            return Err(Error::unsupported("engine", F3_RS_VERBS));
        }
        // **THE MIXED ROW IS SERVED NOW** (design §6's "fused collapse",
        // survey §9's last bullet; wave F3b built the 2R-segment split). What
        // is left is the one thing about it a lane CAN check about itself:
        // a boundary is a position among this fire's own rows, so a host
        // stated one past them names a token this fire does not carry. A
        // device-stated one is clamped by the shell at compose and cannot be
        // checked here at all — which is `FoldLen`'s own rule.
        if let RsVerb::Buffer {
            fold: FoldLen::Host(fold),
            ..
        } = &self.rs
            && *fold > self.rows()
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} folds {fold} of the {} rows it carries",
                self.slot,
                self.rows()
            )));
        }
        Ok(())
    }
}

/// The verb spelling for a channel prediction this engine cannot check.
const F3_CHANNEL_TICKETS: &str =
    "Lane::channels against an engine without the pull-validate and commit-bump kernels";

/// The verb spelling for the recurrent verbs an engine has no device half for.
const F3_RS_VERBS: &str = "Lane::rs beyond RsVerb::Fold (wave F3: the RS device half)";

/// **What an engine states it will actually honour**, carried into
/// [`Lane::validate_for`].
///
/// **A STATED-AND-IGNORED FIELD IS THE ONE FAILURE A TYPED CONTRACT EXISTS TO
/// PREVENT** (design §1 article 3). Two fields of [`Lane`] are predictions an
/// engine either checks or cannot: the channel tickets and the recurrent verb.
/// Neither is a fact about the lane, so neither can be checked by the lane
/// alone — the engine answers, and a lane that asked for something this engine
/// would drop is refused BY NAME instead of quietly served as something else.
///
/// It is a struct rather than two positional `bool`s because it grew from one
/// to two in a single wave and would have grown a third silently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Serves {
    /// This engine advances its channel rings on the device
    /// ([`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit)).
    pub device_channel_commit: bool,
    /// This engine serves [`RsVerb::Buffer`] and [`RsVerb::FoldBuffered`]
    /// ([`Capabilities::rs_verbs`](crate::Capabilities::rs_verbs)).
    pub rs_verbs: bool,
}

impl Serves {
    /// The shape every engine had before F2a: neither prediction honoured.
    pub const NONE: Serves = Serves {
        device_channel_commit: false,
        rs_verbs: false,
    };
}

/// **Whether a lane's recurrent slot begins here** (alto survey §9).
///
/// A recurrent slot IS its history: opening a sequence in a slot another
/// sequence used means zeroing what that one left, because the scan reads the
/// whole bank on its first step. Who KNOWS a slot is fresh is the question
/// this type answers, and until F3 the answer was the wrong store's — the
/// shells keyed the zeroing on `KvDelta::held == 0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsReset {
    /// **The caller states nothing, so the engine keeps its old rule**: a lane
    /// arriving with `kv.held == 0` is a sequence beginning. Exactly the
    /// behaviour every caller had before this field existed, which is why it
    /// is the default.
    #[default]
    Inferred,
    /// The RS store classified this slot as FRESH: zero its banks before the
    /// fire reads them, whatever the KV side says.
    Fresh,
    /// The RS store classified this slot as CONTINUING: leave its banks
    /// alone, whatever the KV side says.
    Held,
}

/// **What this lane's pass does to its recurrent state** (alto design §6).
///
/// Dev's programming model, typed. The default is the only shape that
/// graph-replays and the only one this tree serves; the other two are the
/// speculation vocabulary the Mamba families need, named here so that the
/// exception register has something to point at and so that asking for one is
/// a refusal rather than a silent fold.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsVerb {
    /// Fold each token into the recurrent state inside the forward. The
    /// default, and a predicated commit like every other durable advance
    /// (article 3).
    #[default]
    Fold,
    /// Scatter the in-projection activations into `pages` and leave the folded
    /// state untouched: a draft whose rejection is pure host bookkeeping.
    Buffer {
        /// **The lane's whole buffer run**, as a LIST of physical buffer-page
        /// slot ids in buffer order: entry `j` holds buffer tokens
        /// `[j * page_tokens, (j + 1) * page_tokens)`, where `page_tokens` is
        /// the engine's kv page size ([`PoolFacts`](crate::caps::PoolFacts)).
        ///
        /// The whole run and not this fire's share of it, because
        /// [`RsVerb::FoldBuffered`] addresses the same run from the same
        /// origin and the two spellings must agree without a second field.
        ///
        /// **A LIST AND NOT A RANGE** (wave F3-tail), which is
        /// [`KvDelta::pages`]'s shape for [`KvDelta::pages`]'s reason: the
        /// runtime's recurrent store allocates buffer pages one at a time and
        /// copies them on write after a fork, so a lane's run is contiguous
        /// only by luck. A range forced the runtime to state a first page and
        /// a count it could not honour, and the list IS the translation the
        /// runtime used to carry beside it — `pages[j]` is the physical slot
        /// logical buffer page `j` currently lives in.
        pages: Vec<u32>,
        /// **Which buffer token this fire's first row lands at.** A pure
        /// append states its current occupancy; a re-drafted window states the
        /// accepted boundary it is re-filling from.
        at: u32,
        /// **How many of this fire's rows this lane also FOLDS** (design
        /// §6's fused collapse).
        ///
        /// **`FoldLen::Host(0)` IS THE PURE SCATTER** — the draft whose
        /// rejection is host bookkeeping, with the folded state untouched.
        /// Anything else is the MIXED ROW: one fire that writes every row
        /// into the buffer AND lands the durable state on row `fold`, so the
        /// next window's speculation begins at the accepted boundary without
        /// a second fire.
        ///
        /// Counted in THIS FIRE'S ROWS and bounded by them: `fold == rows`
        /// is the fire that buffers a window and folds all of it, and a
        /// boundary strictly inside the row is what takes the engine's
        /// 2R-segment split (wave F3b).
        fold: FoldLen,
    },
    /// Replay the buffer through conv+recurrence, truncated at the accepted
    /// boundary — the batch fold, skipping the in-projection GEMM.
    FoldBuffered {
        /// The lane's buffer run, addressed exactly as [`RsVerb::Buffer`]
        /// wrote it: the same list of physical page slot ids, page-major from
        /// buffer token zero.
        pages: Vec<u32>,
        /// **Which buffer token the replay starts at** — [`RsVerb::Buffer`]'s
        /// `at`, from the same origin (wave F3b).
        ///
        /// **THE GAP F3 DOCUMENTED AND DID NOT CLOSE.** A fold absorbs tokens
        /// off the FRONT of a lane's buffer but can only release whole
        /// covered pages, so a fold that lands mid-page leaves the survivors
        /// physically offset inside their first page — and a replay that
        /// started at buffer token zero would fold the absorbed tokens a
        /// second time before it reached the live ones. The runtime knew the
        /// number (`buffer_heads[r]`) and had nowhere to state it, so every
        /// fold this tree served had to take the buffer whole or land on a
        /// page boundary. It states it here.
        at: u32,
        /// The host's upper bound on the accepted length, which is what sizes
        /// the launch. The device clamps to it.
        bound: u32,
        /// The accepted length itself.
        len: FoldLen,
    },
}

/// **How long a fold is, and who knows it** (dev ABI v24's `FOLD_LEN_DEVICE`,
/// promoted from a sentinel to a type).
///
/// The accepted count of a speculative pass is device data — it is computed by
/// the verifier on the stream — and article 3 forbids round-tripping it
/// through the host. A shell resolves the port at compose, clamps it to the
/// verb's `bound`, and nothing downstream may branch on which variant it was:
/// the two spellings must produce the same launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldLen {
    /// The host knows it: a fixed count it computed itself.
    Host(u32),
    /// The device knows it: read this descriptor port at compose time.
    Device(eta_ir::registry::Port),
}

/// Where a guest program runs relative to the immutable graph.
///
/// Two points, and there is no third (design §9): guest computation attaches
/// **only** before and after. A mid-graph hook would tear the recorded graph at
/// every layer, which is what the axis mechanism exists to make unnecessary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Boundary {
    /// Before the graph: token preparation, channel reads, state.
    Prologue,
    /// After the graph: sampling, decode logic, channel commit.
    Epilogue,
}

/// One guest-program instance attached to this fire.
///
/// **ONE ATTACHMENT PER INSTANCE PER FIRE, AND [`Attachment::at`] SAYS WHERE
/// THE WHOLE PROGRAM RUNS.** A guest program's `Stage::Prologue` and
/// `Stage::Epilogue` bodies are not two attachments: they are one pass with
/// one readiness gate and one pass-atomic commit, and an engine fires all of a
/// program's stages together. So this names an instance and the side of the
/// graph its pass runs on — a program that reads
/// [`IntrinsicId::Logits`](eta_ir::op::IntrinsicId) must be
/// [`Boundary::Epilogue`], because before the graph there is no readout to
/// point it at. Naming one instance twice in one submission would run its
/// pass twice and commit its channels twice, and an engine refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    /// Which lane of this submission it runs for. The lane whose readout row
    /// an epilogue's `logits` intrinsic is pointed at.
    pub lane: u32,
    /// Which bound instance.
    pub instance: InstanceId,
    /// Which end of the graph.
    pub at: Boundary,
}

/// **One forward pass over the assembled batch — one STEP of a frame.**
///
/// This is what `FireSubmission` was, under the name the execution plane gives
/// it (alto design §2). A frame is 1..=k of these, sealed in order; the
/// degenerate one-step frame is exactly the fire that used to be the contract's
/// unit of work.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    /// The requests in this fire, in submission order. Answers come back in
    /// the same order, whatever order the fire ran them in.
    pub lanes: Vec<Lane>,
    /// The guest programs attached at this fire's boundaries.
    pub attachments: Vec<Attachment>,
}

impl Step {
    /// How many token rows the whole fire carries.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.lanes.iter().map(Lane::rows).sum()
    }

    /// Is this a submission the contract describes?
    ///
    /// Per-lane arithmetic plus the two things that are about the batch: no
    /// slot appears twice, and every attachment names a lane that exists.
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
            // One pass per instance per fire: see [`Attachment`]. A second
            // one would re-run the readiness gate against cursors the first
            // already advanced, and commit the same channel effects twice.
            if self.attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attachment.instance)
            {
                return Err(Error::Invalid(format!(
                    "instance {} is attached twice to one fire, at attachment {index};                      a program's stages are one pass with one commit",
                    attachment.instance
                )));
            }
        }
        Ok(())
    }
}

/// **The unit of work the contract admits: 1..=k steps, sealed in order.**
///
/// The one forward verb takes this and nothing else (alto design §2). Three
/// articles are why it is a frame rather than a fire:
///
/// * **Article 4 (static admission).** Every step is validated, the union of
///   their demands is taken, and it is committed ONCE. `Exhausted` and
///   `Impossible` return with zero side effects; past the commit, stream work
///   is success-only. A per-step admission could refuse step 3 of a frame
///   whose steps 1 and 2 had already written KV, which is a partial frame the
///   caller cannot retry and cannot undo.
/// * **Article 1 (saturation).** `submit` enqueues all k steps before it
///   returns, so step W+1 is on the stream before step W completes. A caller
///   that fired k times could not have been ahead of the device by
///   construction.
/// * **Article 2 (untouched transition).** Nothing host-side stands between
///   consecutive steps — no read, no decision, no synchronize. The steps are
///   handed over together precisely so that no host loop can creep between
///   them.
///
/// **F1 SHIPS THE SHAPE, NOT YET THE PHYSICS.** The shells in this tree run a
/// frame's steps back to back, synchronously, and a k-step frame costs exactly
/// what k fires cost. What changed is that the seam where the saturation goes
/// now exists and the runtime speaks through it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSubmission {
    /// The steps, in the order the device runs them. At least one.
    pub steps: Vec<Step>,
}

impl FrameSubmission {
    /// The degenerate one-step frame — the fire that `fire` used to be.
    #[must_use]
    pub fn of(step: Step) -> FrameSubmission {
        FrameSubmission { steps: vec![step] }
    }

    /// How many token rows the whole frame carries, across its steps.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.steps.iter().map(Step::rows).sum()
    }

    /// Is this a frame the contract describes?
    ///
    /// Article 4's first half: **every** step is checked before any of them is
    /// admitted, which is what makes a refusal free.
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
    /// answer about channel tickets down to [`Lane::validate_for`].
    ///
    /// **AN ENGINE CALLS THIS ONE**, with its own
    /// [`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit),
    /// because whether a stated prediction will be honoured is a fact about
    /// the engine and not about the frame.
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

/// **The receipt for one admitted frame**, one entry per step.
///
/// Synchronous shells fill every step's readouts before `submit` returns, which
/// is what F1 preserves. An asynchronous one answers the id with empty
/// readouts and the runtime's broker correlates the completion on
/// [`FrameTicket::id`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameTicket {
    /// This frame's id, unique for the life of the load.
    pub id: FrameId,
    /// One receipt per submitted step, in submission order.
    pub steps: Vec<FireTicket>,
}

/// One attention layer's captured mass, for one lane (design §9, palo C4).
///
/// **THE LSE, NOT A SCORE MATRIX, AND THAT IS THE HONEST EXPORT.** A full
/// `[query, key]` score matrix is not a value a paged attention kernel ever
/// materializes — it is streamed tile by tile and never exists whole. The
/// log-sum-exp is what the kernel DOES hand back beside `o`; it is the
/// normalizer every per-key score is a ratio against, and it is the quantity a
/// capture consumer can actually be given without the graph growing a second
/// attention.
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
    ///
    /// **THE READOUT IS THE SCORE DOOR, AND THE CHOICE WAS BETWEEN THREE**
    /// (palo C4b). Design §9 prefers the guest-at-the-boundary route — an
    /// intrinsic the way [`Boundary::Epilogue`] binds
    /// `IntrinsicId::Logits` — and that route is genuinely shut for this
    /// value: the intrinsic that exists for scores
    /// ([`IntrinsicId::AttnScore`](eta_ir::op::IntrinsicId)) is registered
    /// at `Stage::OnAttn`, a MID-GRAPH tap, and §9 abolished the third
    /// boundary. It also promises the wrong numbers — `[num_heads, kv_len]`
    /// per-key softmax weights, not a per-query mass — so pointing it at this
    /// column would be a lie that computes. The other candidate, a
    /// [`Pool`](crate::transfer::Pool) variant and a copy verb, is the wrong
    /// noun: a pool is pages a caller resizes and maps, and this is an arena
    /// rectangle the carve placed. What is left is the door the trunk's logits
    /// already take when no guest is attached — this one — and it costs the
    /// contract one field that is empty in every fire nobody captured.
    ///
    /// An epilogue-legal score intrinsic would bind against exactly the
    /// rectangle this field is read from, so nothing here is in that route's
    /// way; what it needs is a row in `eta-ir`'s intrinsic table, which is
    /// the frozen side of this wave's seam.
    #[serde(default)]
    pub scores: Vec<LayerScores>,
}

/// The receipt for one accepted fire.
///
/// **This is all that is left of `completion.rs`.** That module was 807 lines
/// of run-ahead machinery — a `CompletionBroker` with a live registry and a
/// recycling pool of terminal cells, a `SubmissionCompletion` future parked on
/// a `waker` slot table, per-work-item leases, a four-state `TerminalOutcome`
/// written through a `#[repr(C)]` atomic cell — living inside the contract
/// that describes what an engine *is*. None of it is: it is how the RUNTIME
/// decides to run ahead of a device, and it belongs beside the scheduler that
/// makes that decision (decision 19). The contract keeps the receipt.
///
/// The shells this contract has are synchronous: the eager walk completes
/// before `fire` returns, and [`FireTicket::readouts`] is already filled. An
/// asynchronous shell answers with the id and an empty readout list, and
/// [`FireTicket::id`] is what the runtime-side broker correlates its completion
/// on.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FireTicket {
    /// This fire's id, unique for the life of the load.
    pub id: FireId,
    /// One entry per submitted lane, in submission order. Empty from an engine
    /// that answers before the device is done.
    pub readouts: Vec<LaneReadout>,
}

/// The `encode` verb's argument: non-text modalities in, embedding rows out.
///
/// Kept whole from `MediaEncodePlan` — it is a batch of independent blobs with
/// an anchor row each, and there is no per-lane structure in it to purify.
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
    /// [`Error::Invalid`] for a payload with no anchor, an anchor with
    /// no payload, or a partition that does not cover its bytes.
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
