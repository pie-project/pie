//! What the runtime assembles a fire out of. The contract's [`Lane`] is the
//! per-request form that crosses the boundary, so nothing here flattens it;
//! what this module holds is the runtime's own bookkeeping about a request —
//! geometry class, outstanding page rewrites, folded recurrent slots — none of
//! which the engine reads.
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
//! A request is lanes, plural: a "prebuilt" request can carry several (the
//! SDK's one-token-per-lane lowering is the standing example), so batch
//! assembly is a concatenation rather than several simultaneous CSR merges.

use engine::fire::{Step, KvDelta, Lane};
use eta_ir::registry::GeometryClass;

use crate::engine::completion::TerminalCell;

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
    /// Does this request's bound instance run a pass at the fire's boundary?
    /// A per-request fact, not a fire-wide policy: a pipeline submit fires a
    /// `BoundForwardPass` attached to a model fire, while everything else (a
    /// prebuilt rider, a hand-built test request) has no such instance.
    /// Defaults to `false`. Which boundary it runs at is not a field —
    /// `batch` always stamps [`Boundary::Epilogue`](engine::fire::Boundary::Epilogue).
    pub boundary_program: bool,
    /// The media spans this request's pass attached, keyed by lane; empty
    /// for every text-only request. `lane` indexes this request's own
    /// [`lanes`](FireRequest::lanes) (a submission can't know which step it
    /// co-batches into); `scheduler::batch` rebases it on concatenation.
    pub media: Vec<engine::fire::StepMedia>,
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
    /// A view, not a field: derived here so a CSR and the lanes it cuts
    /// cannot disagree.
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
    /// Arithmetic over the lane's own `held` and rows, computed rather than carried.
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

/// One step of a frame, as the engine lane fires it. A step is one fire: the
/// per-request state lives on the lanes, so a step carries only the submission
/// plus the runtime's own bookkeeping about it.
pub struct StepFire {
    /// What crosses the boundary. Moved out at submission, not copied:
    /// `fire_frame` takes every step's submission into the one
    /// `FrameSubmission` the engine is handed, so this is empty from the
    /// instant the frame reaches the device.
    pub submission: Step,
    /// The terminal cell each lane's work item settles through, parallel to
    /// [`Step::lanes`]. The engine answers a `Result<FrameTicket>` (one
    /// `FireTicket` per step), and [`crate::engine::completion::settle`]
    /// writes these from it.
    pub terminal_cells: Vec<*mut TerminalCell>,
    /// Which bound instance each lane belongs to, parallel to the lanes.
    /// `batch` builds [`Step::attachments`](engine::Step) out of exactly
    /// this vector; it also stays here because the scheduler's own
    /// in-flight tables read the association for lanes with no attachment,
    /// and the engine lane pumps a bound instance's channels by id.
    pub instances: Vec<u64>,
    /// Each lane's logical fire id, for the log and the watchdog.
    pub logical_fire_ids: Vec<u64>,
}

/// One sealed frame: its steps, in order.
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

/// Expand every lane's mask into bitmap words. `request_indptr` cuts
/// `word_indptr` per lane because a lane can carry more than one mask (a
/// windowed prefill states one restriction per query row). The causal bound
/// is not folded in here — this is the run encoding expanded; every consumer
/// intersects the bits with the row's own bound.
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
            // Stamped only where a page reference is the guest's to resolve;
            // every other lane's pages are pool ids by the time they reach here.
            translation: Vec::new(),
        },
        ..Lane::default()
    }
}
