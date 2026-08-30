//! What the runtime assembles a fire out of.
//!
//! # The CSRs were never a submission
//!
//! This module replaces `engine::plan::LaunchPlan` — sixty-two fields,
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

use engine::fire::{Step, KvDelta, Lane};
use eta_ir::registry::GeometryClass;

use crate::engine::completion::TerminalCell;

// **`palo B-rs` CLOSED, AND THE VERB CROSSES NOW** (alto wave F3-tail). An
// `RsPlan` struct stood here: eleven parallel `Vec<u32>` arms — `slot_ids`,
// `slot_flags`, `fold_lens`, `buffer_slot_ids` + its CSR, `buffer_read_*`,
// `buffer_heads`, `translation` + its CSR — the last survivors of
// `LaunchPlan`'s recurrent half, carried on `FireRequest::rs` because the
// contract had no recurrent field at all and the runtime's own store was
// built on them. Nothing across the boundary could read one, so a lane that
// meant to scatter a speculative window into a buffer was submitted as an
// ordinary fold and the device folded it.
//
// `engine::RsVerb` and `engine::RsReset` are that vocabulary, and
// they are fields of the LANE — so the arms have nowhere left to be:
// `pipeline::fire::rs::PreparedRs::apply_to` stamps one verb and one reset
// fact onto the lane that carries each row, `RsVerb::Buffer::pages` IS the
// translation the ninth and tenth arms carried, and everything else was
// either the runtime store's own bookkeeping (which never left
// `PreparedRs`) or a second spelling of a number the verb states once
// (article 8).

// **`palo B-media` CLOSED THE OTHER WAY** (alto E). A `Media` struct stood
// here — twenty fields of pixel bytes, patch grids, mRoPE positions, audio
// features and precomputed embedding rows, kept so the runtime would have
// somewhere to hold an encode's payload. Nothing ever wrote one. Every
// `FireRequest` built in this tree left it `Default`, so
// `offload::try_encode`'s `media.is_empty()` gate was true on every fire and
// the encode seam was dead in front of a payload that did not exist. The
// payload comes back with the verb that produces it
// (`engine::MediaEncode` is `encode`'s argument, and what a fire needs
// afterwards is rows in the arena — a seam the shell resolves).
//
// A `ChannelTicket` struct stood here too: a channel's expected ring cursors,
// stated on the REQUEST as `FireRequest::tickets` beside a
// `device_channel_tickets` flag that said whether the engine had a device
// half to check them, which `scheduler::batch` then transcribed onto the
// attached lane. Both are gone. The reservation is stamped straight onto the
// lane that carries the pass (`engine::Lane::channels`) by
// `pipeline::fire`'s `TicketReservation::apply_to` — the party that mints it
// and the party that knows whether the instance's channels were adopted. Two
// spellings of one number is what article 8 forbids.

/// One request, as the runtime holds it between submit and fire.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FireRequest {
    /// **What crosses the boundary.** One entry per row group the request
    /// fires; the batch's [`Step::lanes`] is these, concatenated.
    pub lanes: Vec<Lane>,
    /// Which geometry class this request fires in — how much of the fire's
    /// descriptor the engine resolves on the device rather than reading out
    /// of the submission. Stated at bind time
    /// ([`InstanceBinding::geometry`](engine::InstanceBinding)); carried
    /// here because the scheduler groups by it.
    pub geometry: GeometryClass,
    /// How many layers to run, for a partial-depth fire.
    pub max_layers: Option<u32>,
    /// Every lane carries exactly one token.
    pub single_token_mode: bool,
    /// The mask came from the guest rather than from the causal derivation.
    pub has_user_mask: bool,
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
    /// [`Step::attachments`](engine::Step) carries
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
/// rewrite for a fork mover no shell in this tree has — both spellings are
/// deleted rather than carried (alto E); `required_kv_pages` was a
/// frame-union high-water the engine used to size an admission check it
/// makes for itself now ([`Error::Exhausted`](engine::Error::Exhausted)
/// carries the numbers).
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
///
/// **`request_indptr` CUTS `word_indptr` PER LANE BECAUSE A LANE CAN CARRY
/// MORE THAN ONE MASK**, and since `Masking::Rows` it actually does: a
/// windowed prefill states one restriction per query row, and each of them
/// expands into its own bitmap. The two-level CSR was already this shape when
/// every lane put exactly one mask in it — `LaunchPlan::bitmask_words` came
/// off a wire form that cut a flat mask vector per request — so a per-row
/// lane fills it rather than changing it. The causal bound is NOT folded in
/// here: this is the run encoding expanded, and every consumer that reads the
/// bits intersects them with the row's own bound (`engine_cuda::mask`).
#[must_use]
pub fn bitmask_words(lanes: &[Lane]) -> MaskWords {
    let mut request_indptr = Vec::with_capacity(lanes.len() + 1);
    let mut word_indptr = vec![0u32];
    let mut words: Vec<u32> = Vec::new();
    request_indptr.push(0);
    for lane in lanes {
        for mask in lane.mask.iter().flat_map(|masking| masking.masks()) {
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
            // Stamped only where a page reference is the guest's to resolve
            // (`pipeline::fire::stamp_lane_translation`); every other lane's
            // pages are pool ids by the time they reach here.
            translation: Vec::new(),
        },
        ..Lane::default()
    }
}

/// Re-exported so a reader of this module does not have to reach two crates
/// deep for the nouns its own signatures are written in.
pub use engine::fire::{Attachment, Boundary, FireTicket, LaneReadout};
