//! What the runtime assembles a fire out of.
//!
//! # The CSRs were never a submission
//!
//! This module replaces `engine_api::plan::LaunchPlan` — sixty-two fields,
//! most of them parallel `Vec<u32>` arms (`qo_indptr`, `kv_page_indptr`,
//! `rs_translation_indptr`, `embed_block_indptr`, `image_mrope_indptr`, …)
//! — and the four-hundred-line `StepSubmission::validate` whose whole job was
//! checking that they were the same length as each other. The runtime
//! flattened its per-request state into eleven CSRs and the engine's first
//! act was to walk them back into per-request form.
//!
//! The contract's [`Lane`] is the per-request form, so the flattening is
//! gone. What survives here is what the CSRs were NOT saying: the
//! runtime's own bookkeeping about a request — which geometry class it fires
//! in, which page rewrite is outstanding, which recurrent slots it folds —
//! none of which is a shape the engine reads.
//!
//! ```text
//!   FireRequest            = the runtime's record of one submitted request
//!     .lanes: Vec<Lane>      ── what crosses the boundary
//!     everything else        ── what does not (each field says where it went)
//!
//!   Step                   = one fire: the lanes of every request in a step
//!   FrameSubmission        = the frame's steps, sealed — what `submit` takes
//! ```
//!
//! # A request is LANES, plural, and always was
//!
//! `qo_indptr` had `rows + 1` entries and a "prebuilt" request could carry
//! several — the SDK's one-token-per-lane lowering is the standing example.
//! So the natural per-request shape is a `Vec<Lane>` rather than one, and
//! batch assembly is a concatenation instead of eleven simultaneous CSR
//! merges (`scheduler::wire`, which was that merge and is now the trim).

use engine_api::fire::{Step, KvDelta, Lane};
use tensor_ir::registry::GeometryClass;

use crate::engine::completion::TerminalCell;

/// The recurrent-state half of a request.
///
/// **`palo B-rs`: NOTHING CARRIES THIS ACROSS THE BOUNDARY YET.** The old
/// `LaunchPlan` had eight parallel arms for it (`rs_slot_ids`,
/// `rs_slot_flags`, `rs_fold_lens`, `rs_buffer_*`, `rs_translation*`) and the
/// new contract has none: a recurrent slot is a `CacheRow::State` seat in the
/// plan, the shell owns the pool, and [`Lane::slot`] is the sequence's seat in
/// BOTH pools. What has no seat yet is the per-fire verb — reset this slot,
/// fold this many positions, read this buffer — which the design puts on a
/// model-declared axis rather than on the submission (§8). The runtime still
/// computes it, because its own recurrent store is built on it and
/// `copy_state` moves what it names; it stops at the boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RsPlan {
    /// The state slot each lane writes.
    pub slot_ids: Vec<u32>,
    /// Per slot: reset / fold / buffer-write / device-fold-length.
    pub slot_flags: Vec<u8>,
    /// Per slot: how many positions the fold covers.
    pub fold_lens: Vec<u32>,
    /// The buffer slots each lane writes.
    pub buffer_slot_ids: Vec<u32>,
    /// `[lanes + 1]`: each lane's span of [`RsPlan::buffer_slot_ids`].
    pub buffer_slot_indptr: Vec<u32>,
    /// The buffer slots each lane reads.
    pub buffer_read_slot_ids: Vec<u32>,
    /// `[lanes + 1]`: each lane's span of [`RsPlan::buffer_read_slot_ids`].
    pub buffer_read_indptr: Vec<u32>,
    /// How much of each read buffer is live.
    pub buffer_read_lens: Vec<u32>,
    /// Each read buffer's ring head.
    pub buffer_heads: Vec<u32>,
    /// Slot ids rewritten since the last fire.
    pub translation: Vec<u32>,
    /// `[lanes + 1]`: each lane's span of [`RsPlan::translation`].
    pub translation_indptr: Vec<u32>,
}

impl RsPlan {
    /// True when this request touches no recurrent state at all — which is
    /// every request of a plain KV model.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slot_ids.is_empty()
            && self.buffer_slot_ids.is_empty()
            && self.buffer_read_slot_ids.is_empty()
            && self.translation.is_empty()
    }
}

/// The non-text half of a request: images, audio, precomputed embedding rows.
///
/// **`palo B-media`: THE ENCODE VERB TAKES THESE, THE FIRE DOES NOT.** The old
/// `LaunchPlan` carried twenty fields of pixel bytes, patch grids, mRoPE
/// positions and anchor rows *inside the fire*, so every decode step shipped a
/// dozen empty vectors. The contract splits it:
/// [`MediaEncode`](engine_api::MediaEncode) is the `encode` verb's argument
/// and answers embedding rows, and what a fire needs afterwards is rows in the
/// arena — which is a seam the shell resolves, not a payload the submission
/// carries. So the runtime keeps the payload here, hands it to `encode`, and
/// the rows-into-a-fire half is unwired.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Media {
    /// Per image: `(temporal, height, width)` patch counts.
    pub image_grids: Vec<u32>,
    /// Where each image's embeddings anchor, as a token position.
    pub image_anchor_positions: Vec<u32>,
    /// Where each image's embeddings anchor, as a row of this request.
    pub image_anchor_rows: Vec<u32>,
    /// Pixel bytes for every image.
    pub image_pixels: Vec<u8>,
    /// Byte offsets splitting [`Media::image_pixels`] per image.
    pub image_pixel_indptr: Vec<u32>,
    /// Per-image mRoPE positions.
    pub image_mrope_positions: Vec<u32>,
    /// Byte offsets splitting [`Media::image_mrope_positions`] per image.
    pub image_mrope_indptr: Vec<u32>,
    /// Per patch: its `(y, x)` position.
    pub image_patch_positions: Vec<u32>,
    /// Feature bytes for every audio clip.
    pub audio_features: Vec<u8>,
    /// Byte offsets splitting [`Media::audio_features`] per clip.
    pub audio_feature_indptr: Vec<u32>,
    /// Where each clip's embeddings anchor, as a row of this request.
    pub audio_anchor_rows: Vec<u32>,
    /// Precomputed embedding rows, `bf16`.
    pub embed_rows: Vec<u8>,
    /// Byte offsets splitting [`Media::embed_rows`] per block.
    pub embed_indptr: Vec<u32>,
    /// Each block's `(rows, width)`.
    pub embed_shapes: Vec<u32>,
    /// Where each block anchors, as a row of this request.
    pub embed_anchor_rows: Vec<u32>,
}

impl Media {
    /// True when the request is text only.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.image_anchor_rows.is_empty()
            && self.audio_anchor_rows.is_empty()
            && self.embed_anchor_rows.is_empty()
    }
}

/// A channel's expected ring cursors at this fire.
///
/// **THE ASSERTION MOVED TO THE PARTY THAT OWNS THE RINGS.** These were
/// `channel_expected_head`/`channel_expected_tail` on the plan and
/// `channel_ticket_indptr` on the step — a per-fire claim, shipped across the
/// boundary, about where a guest program's rings stand. The engine answers
/// that question for itself now: it gates every attached instance's readiness
/// against its OWN cursors before it launches anything
/// (`engine_cuda::program::Plane::ready`), and a ring that is not where the
/// caller thought is `Exhausted` rather than a mismatch nobody checked. So
/// nothing carries these across; they stay as the runtime's own run-ahead
/// reservation, which is the half [`crate::pipeline::fire`]'s
/// `TicketReservation` was always about — frame validation at k > 1 reads
/// `device_ring_backlog` and `writer_available_cells`, and both are counted
/// from the reservations rather than from this vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChannelTicket {
    /// Which channel this predicts about, by its GLOBAL id — the only
    /// spelling an engine and the runtime share, since an instance numbers
    /// its channels densely and the runtime numbers them by registration.
    pub channel: u64,
    /// The head the reader expects.
    pub head: u64,
    /// The tail the writer expects.
    pub tail: u64,
}

/// One request, as the runtime holds it between submit and fire.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FireRequest {
    /// **What crosses the boundary.** One entry per row group the request
    /// fires; the batch's [`Step::lanes`] is these, concatenated.
    pub lanes: Vec<Lane>,
    /// Which geometry class this request fires in — how much of the fire's
    /// descriptor the engine resolves on the device rather than reading out
    /// of the submission. Stated at bind time
    /// ([`InstanceBinding::geometry`](engine_api::InstanceBinding)); carried
    /// here because the scheduler groups by it.
    pub geometry: GeometryClass,
    /// The engine reads this request's geometry from a channel the device
    /// wrote, so the host's copy of it is advisory.
    pub device_resolved_geometry: bool,
    /// This request's attention mask lives in a device cell, which is what
    /// makes it un-batchable with another program's.
    pub dense_device_mask: bool,
    /// The bound program writes `attn_page_mask`.
    pub hook_page_mask: bool,
    /// The highest page id this request's geometry reaches, plus one.
    pub required_kv_pages: u32,
    /// Page ids rewritten since this request's last fire, parallel to the
    /// lane's own page list.
    pub kv_translation: Vec<u32>,
    /// Which mapping version [`FireRequest::kv_translation`] is from.
    pub kv_translation_version: u64,
    /// The byte span this request promises to write inside, when it resolves
    /// its own write targets on the device.
    pub kv_write_bounds: Option<(u64, u64)>,
    /// How many layers to run, for a partial-depth fire.
    pub max_layers: Option<u32>,
    /// Every lane carries exactly one token.
    pub single_token_mode: bool,
    /// The mask came from the guest rather than from the causal derivation.
    pub has_user_mask: bool,
    /// The recurrent-state half. See [`RsPlan`].
    pub rs: RsPlan,
    /// The non-text half. See [`Media`].
    pub media: Media,
    /// The guest program's expected ring cursors. See [`ChannelTicket`].
    pub tickets: Vec<ChannelTicket>,
    /// **Does the engine this request is bound for validate predictions on
    /// the device?** (alto design §1 article 3, wave F2a.)
    ///
    /// True exactly when every one of this instance's channels was ADOPTED —
    /// the engine published the pinned host half and its control kernels read
    /// it — which is the same fact
    /// [`Capabilities::device_channel_commit`](engine_api::Capabilities::device_channel_commit)
    /// states, asked per channel where the runtime already knows it. Only
    /// then do the reservations above cross as
    /// [`Lane::channels`](engine_api::Lane): an engine with no device half to
    /// check them refuses a stated prediction by name rather than ignoring it.
    pub device_channel_tickets: bool,
    /// **Does this request's bound instance run a pass at the fire's
    /// boundary?** (`palo B2`, design §9.)
    ///
    /// It is a per-request FACT rather than a fire-wide policy because the
    /// two ways a request reaches the engine lane are genuinely different: a
    /// pipeline submit fires a `BoundForwardPass`, which IS a guest program
    /// attached to a model fire, and everything else — a prebuilt rider, a
    /// test that builds a `FireRequest` by hand — has no instance whose
    /// channels the engine carved. Defaulting to `false` is what makes the
    /// second kind byte-identical to what it was: an empty
    /// [`Step::attachments`] is the same submission it always was,
    /// down to the shell taking the same branch.
    ///
    /// Which BOUNDARY it runs at is not a field: a program's stages are one
    /// pass with one commit, and a pass that reads the fire's logits can only
    /// be the epilogue ([`Attachment`]'s own doc). `batch` stamps
    /// [`Boundary::Epilogue`].
    pub boundary_program: bool,
}

impl FireRequest {
    /// A single-lane request over `tokens`.
    #[must_use]
    pub fn one(lane: Lane) -> FireRequest {
        FireRequest {
            lanes: vec![lane],
            ..FireRequest::default()
        }
    }

    /// How many lanes this request contributes to a fire.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.lanes.len()
    }

    /// How many token rows it contributes.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.lanes.iter().map(|lane| lane.tokens.len()).sum()
    }

    /// Every page id its lanes name.
    pub fn pages(&self) -> impl Iterator<Item = u32> + '_ {
        self.lanes.iter().flat_map(|lane| lane.kv.pages.iter().copied())
    }

    /// The first lane, for the single-lane requests that are almost all of
    /// them.
    #[must_use]
    pub fn lane(&self) -> Option<&Lane> {
        self.lanes.first()
    }

    /// The first lane, mutably.
    pub fn lane_mut(&mut self) -> Option<&mut Lane> {
        self.lanes.first_mut()
    }

    /// The token-row CSR over this request's lanes — `[lanes + 1]` entries.
    ///
    /// **A VIEW, NOT A FIELD.** `qo_indptr` was a stored arm of the wire plan
    /// and half the runtime read it; it is derived here because a CSR and the
    /// lanes it cuts cannot disagree if only one of them exists. The readers
    /// that still want the flat shape — the recurrent planner, the geometry
    /// validators — ask for it.
    #[must_use]
    pub fn qo_indptr(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.lanes.len() + 1);
        let mut at = 0u32;
        out.push(0);
        for lane in &self.lanes {
            at = at.saturating_add(lane.rows());
            out.push(at);
        }
        out
    }

    /// How full the last lane's last KV page is once this request has fired.
    ///
    /// Was `kv_last_page_lens`, an arm of the wire plan with one entry per
    /// lane, of which the scheduler read exactly the last. It is arithmetic
    /// over the lane's own `held` and rows, so it is computed rather than
    /// carried.
    #[must_use]
    pub fn last_page_len(&self, page_size: u32) -> u32 {
        if page_size == 0 {
            return 0;
        }
        let Some(lane) = self.lanes.last() else {
            return 0;
        };
        let after = lane.kv.held.saturating_add(lane.rows());
        match after % page_size {
            0 if after == 0 => 0,
            0 => page_size,
            rest => rest,
        }
    }
}

/// One step of a frame, as the engine lane fires it.
///
/// **A STEP IS ONE FIRE NOW.** `StepSubmission` carried a `LaunchPlan` plus
/// eleven side tables — `roster_rows`, `sub_batch_indptr`,
/// `sub_batch_class`, `program_row_indptr`, `channel_ticket_indptr`,
/// `region_row_indptr`, `region_sig`, `region_k`, … — every one of them a
/// mapping from the flattened CSRs back to the per-request state they were
/// flattened from. Lanes are that state, so the tables have nothing left to
/// map and what a step carries is the submission plus the runtime's own
/// bookkeeping about it.
pub struct StepFire {
    /// What crosses the boundary.
    ///
    /// **MOVED OUT AT SUBMISSION, NOT COPIED.** `fire_frame` takes every
    /// step's submission into the one `FrameSubmission` the engine is handed
    /// (alto design §2), so this is empty from the instant the frame reaches
    /// the device — everything the runtime still needs after that point is the
    /// two fields below, which is why they are fields rather than lookups
    /// into the submission.
    pub submission: Step,
    /// The terminal cell each lane's work item settles through, parallel to
    /// [`Step::lanes`]. Runtime-side: the engine answers a
    /// `Result<FrameTicket>` — one `FireTicket` per step — and
    /// [`crate::engine::completion::settle`] writes these from it.
    pub terminal_cells: Vec<*mut TerminalCell>,
    /// Which bound instance each lane belongs to, parallel to the lanes.
    ///
    /// This is what
    /// [`Step::attachments`](engine_api::Step) carries
    /// for every lane whose request set
    /// [`FireRequest::boundary_program`] — `batch` builds the attachments out
    /// of exactly this vector. It stays here as well because the scheduler's
    /// own in-flight tables read the association for lanes that carry no
    /// attachment too, and because the engine lane pumps a bound instance's
    /// channels by id.
    pub instances: Vec<u64>,
    /// Each lane's logical fire id, for the log and the watchdog.
    pub logical_fire_ids: Vec<u64>,
}

/// One sealed frame: its steps, in order.
///
/// `FrameSubmission`'s four frame-level fields are gone with the wire form.
/// `instance_ids` was the roster the step tables indexed;
/// `kv_translation`/`kv_translation_indptr` were a per-roster-lane page
/// rewrite, which is [`KvDelta::translation`](engine_api::KvDelta) on the
/// lane that owns it; `required_kv_pages` was a frame-union high-water the
/// engine used to size an admission check it makes for itself now
/// ([`Error::Exhausted`](engine_api::Error::Exhausted) carries
/// the numbers).
#[derive(Default)]
pub struct FrameFire {
    /// The steps, in the order the lane fires them.
    pub steps: Vec<StepFire>,
}

impl FrameFire {
    /// Every terminal cell this frame owns, across its steps.
    pub fn terminal_cells(&self) -> impl Iterator<Item = *mut TerminalCell> + '_ {
        self.steps
            .iter()
            .flat_map(|step| step.terminal_cells.iter().copied())
    }
}

/// The masks of one request, expanded into the bitmap a kernel reads.
///
/// Kept from `LaunchPlan::bitmask_words` — the shells that want a dense
/// bitmap rather than runs still want one, and the expansion is the same.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaskWords {
    /// `[lanes + 1]`: each lane's span of [`MaskWords::word_indptr`].
    pub request_indptr: Vec<u32>,
    /// `[masks + 1]`: each mask's span of [`MaskWords::words`].
    pub word_indptr: Vec<u32>,
    /// The bits, one `u32` per 32 positions.
    pub words: Vec<u32>,
}

/// Expand every lane's mask into bitmap words.
#[must_use]
pub fn bitmask_words(lanes: &[Lane]) -> MaskWords {
    let mut request_indptr = Vec::with_capacity(lanes.len() + 1);
    let mut word_indptr = vec![0u32];
    let mut words: Vec<u32> = Vec::new();
    request_indptr.push(0);
    for lane in lanes {
        if let Some(mask) = &lane.mask {
            let start = words.len();
            words.resize(start + mask.words(), 0);
            mask.expand_into(&mut words[start..]);
            word_indptr.push(u32::try_from(words.len()).unwrap_or(u32::MAX));
        }
        request_indptr.push(u32::try_from(word_indptr.len() - 1).unwrap_or(u32::MAX));
    }
    MaskWords {
        request_indptr,
        word_indptr,
        words,
    }
}

/// A lane with `tokens` fed into `slot`, everything else defaulted.
#[must_use]
pub fn lane_of(slot: u32, tokens: Vec<u32>, held: u32, pages: Vec<u32>) -> Lane {
    Lane {
        slot,
        tokens,
        kv: KvDelta {
            held,
            pages,
            translation: Vec::new(),
        },
        ..Lane::default()
    }
}

/// Re-exported so a reader of this module does not have to reach two crates
/// deep for the nouns its own signatures are written in.
pub use engine_api::fire::{Attachment, Boundary, FireTicket, LaneReadout};
