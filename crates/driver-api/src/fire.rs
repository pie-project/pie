//! The fire submission — one forward pass over the batch the engine assembled.
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
//! `Classify::of(&Request)` computed engine-side, and it is what
//! `driver::fire::compose` turns into a class, and therefore into the row
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
//! submission: the engine flattened its per-request state into eleven CSRs,
//! and the driver's first act was to walk them back into per-request form.
//!
//! Here a lane is a lane. The CSRs are the SHELL's — it builds them in
//! `compose`, from a `Vec<Lane>` — and a whole class of "these two arms
//! disagree" failures cannot be submitted. What is left to validate is
//! arithmetic about one lane at a time.

use serde::{Deserialize, Serialize};

use crate::error::{DriverError, Result};
use crate::program::InstanceId;

/// A fire's id, minted by the driver, unique for the life of a load.
pub type FireId = u64;

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
    /// engine keeping the page table, which is what a load with an exported
    /// [`KvHandle`](crate::transfer::KvHandle) and a remote peer writing into
    /// it needs.
    pub pages: Vec<u32>,
    /// Rewritten page ids, one per entry of `pages`, when a fork moved this
    /// lane's pages since the last fire. Empty when nothing moved.
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

/// One request inside a fire, as the engine submits it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lane {
    /// Which pool slot this request's sequence lives in.
    pub slot: u32,
    /// **The lane's fact bits.** `Cond::Fact(bit)` indexes them; the model's
    /// own `Classify::of` computed them (decision 18). A word this load's
    /// artifact has no class for is a refusal, not a default: it means the
    /// engine and the shell disagree about what is loaded.
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
    pub mask: Option<Mask>,
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
    /// (`driver_cuda::Fault::DraftWord`), which is [`Lane::mask`]'s rule and
    /// [`Lane::adapter`]'s rule for the third time.
    ///
    /// **THE ROW ALIGNMENT IS THE CALLER'S** (`model::qwen_3`'s own note): the
    /// head was trained on `(hidden at p, token at p+1)` and is fed `(x, tok)`
    /// at one row, so a drafting lane's row `r` must carry the token one
    /// position past the hidden the trunk leaves at `r`. The contract states
    /// the requirement; no driver can check it.
    pub drafts: bool,
    /// **Keep this lane's attention mass** (design §9's score-capture
    /// archetype, palo C4). `true` is what makes the lane's `captures_scores`
    /// fact true, and what puts its per-layer log-sum-exp into
    /// [`LaneReadout::scores`].
    ///
    /// [`Lane::drafts`]'s twin in every respect, including the refusal
    /// (`driver_cuda::Fault::ScoreWord`) and including being a bare boolean:
    /// what the capture arm needs is the bit, and everything else about the
    /// observation — which layers, how wide, how many rows — is the model
    /// text's to declare and the artifact's to carry.
    pub captures_scores: bool,
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
    /// [`DriverError::Invalid`] when the positions do not match the tokens, or
    /// a readout names a row the lane does not have.
    pub fn validate(&self) -> Result<()> {
        if !self.positions.is_empty() && self.positions.len() != self.tokens.len() {
            return Err(DriverError::Invalid(format!(
                "lane in slot {} has {} positions for {} tokens",
                self.slot,
                self.positions.len(),
                self.tokens.len()
            )));
        }
        if let Readout::Rows(rows) = &self.readout
            && let Some(&row) = rows.iter().find(|&&r| r >= self.rows())
        {
            return Err(DriverError::Invalid(format!(
                "lane in slot {} reads row {row} of the {} it has",
                self.slot,
                self.rows()
            )));
        }
        if !self.kv.translation.is_empty() && self.kv.translation.len() != self.kv.pages.len() {
            return Err(DriverError::Invalid(format!(
                "lane in slot {} translates {} of its {} pages",
                self.slot,
                self.kv.translation.len(),
                self.kv.pages.len()
            )));
        }
        Ok(())
    }
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
/// one readiness gate and one pass-atomic commit, and a driver fires all of a
/// program's stages together. So this names an instance and the side of the
/// graph its pass runs on — a program that reads
/// [`IntrinsicId::Logits`](tensor_ir::op::IntrinsicId) must be
/// [`Boundary::Epilogue`], because before the graph there is no readout to
/// point it at. Naming one instance twice in one submission would run its
/// pass twice and commit its channels twice, and a driver refuses it.
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

/// One forward pass over the assembled batch.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireSubmission {
    /// The requests in this fire, in submission order. Answers come back in
    /// the same order, whatever order the fire ran them in.
    pub lanes: Vec<Lane>,
    /// The guest programs attached at this fire's boundaries.
    pub attachments: Vec<Attachment>,
}

impl FireSubmission {
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
    /// [`DriverError::Invalid`] with the first thing that is wrong.
    pub fn validate(&self) -> Result<()> {
        if self.lanes.is_empty() {
            return Err(DriverError::Invalid("a fire carries no lanes".into()));
        }
        for (index, lane) in self.lanes.iter().enumerate() {
            lane.validate()?;
            if self.lanes[..index].iter().any(|l| l.slot == lane.slot) {
                return Err(DriverError::Invalid(format!(
                    "slot {} appears twice in one fire, at lane {index}",
                    lane.slot
                )));
            }
        }
        let lanes = u32::try_from(self.lanes.len()).unwrap_or(u32::MAX);
        for (index, attachment) in self.attachments.iter().enumerate() {
            if attachment.lane >= lanes {
                return Err(DriverError::Invalid(format!(
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
                return Err(DriverError::Invalid(format!(
                    "instance {} is attached twice to one fire, at attachment {index};                      a program's stages are one pass with one commit",
                    attachment.instance
                )));
            }
        }
        Ok(())
    }
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
    /// ([`IntrinsicId::AttnScore`](tensor_ir::op::IntrinsicId)) is registered
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
    /// way; what it needs is a row in `tensor-ir`'s intrinsic table, which is
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
/// that describes what a driver *is*. None of it is: it is how the ENGINE
/// decides to run ahead of a device, and it belongs beside the scheduler that
/// makes that decision (decision 19). The contract keeps the receipt.
///
/// The shells this contract has are synchronous: the eager walk completes
/// before `fire` returns, and [`FireTicket::readouts`] is already filled. An
/// asynchronous shell answers with the id and an empty readout list, and
/// [`FireTicket::id`] is what the engine-side broker correlates its completion
/// on.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FireTicket {
    /// This fire's id, unique for the life of the load.
    pub id: FireId,
    /// One entry per submitted lane, in submission order. Empty from a driver
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
    /// The embedding rows, `bf16`, filled by the driver.
    pub output_rows: Vec<u8>,
    /// Byte offsets splitting `output_rows` per image then per clip.
    pub output_row_indptr: Vec<u32>,
}

impl MediaEncode {
    /// Is this an encode the contract describes?
    ///
    /// # Errors
    ///
    /// [`DriverError::Invalid`] for a payload with no anchor, an anchor with
    /// no payload, or a partition that does not cover its bytes.
    pub fn validate(&self) -> Result<()> {
        const F32: usize = size_of::<f32>();
        const U16: usize = size_of::<u16>();
        let bad = |why: String| Err(DriverError::Invalid(why));

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
        return Err(DriverError::Invalid(format!(
            "{name} starts at {:?}, not 0",
            indptr.first()
        )));
    }
    if indptr.last().copied() != u32::try_from(bytes).ok() {
        return Err(DriverError::Invalid(format!(
            "{name} ends at {:?}, not the {bytes} bytes it partitions",
            indptr.last()
        )));
    }
    for w in indptr.windows(2) {
        let ordered = if strict { w[0] < w[1] } else { w[0] <= w[1] };
        if !ordered {
            return Err(DriverError::Invalid(format!(
                "{name} segment {}..{} is empty or inverted",
                w[0], w[1]
            )));
        }
        if !(w[0] as usize).is_multiple_of(align) || !(w[1] as usize).is_multiple_of(align) {
            return Err(DriverError::Invalid(format!(
                "{name} segment {}..{} is not {align}-byte aligned",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}
