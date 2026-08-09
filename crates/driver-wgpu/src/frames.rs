//! A frame from the engine, turned into fires.
//!
//! # The two allocators, and which one wins here
//!
//! This driver has a page allocator of its own -- [`crate::pages::Book`] --
//! and it is the right one for a server built on this crate alone. It is the
//! WRONG one for the engine, whose scheduler already owns eviction, prefix
//! sharing and the copy plans that move pages between conversations, and
//! which hands a driver a `kv_page_indices` CSR naming physical pages it
//! chose.
//!
//! Two allocators handing out page 7 is not an error anyone sees: attention
//! reads another conversation's keys and the model answers fluently. So this
//! path does not touch the book, and [`crate::turns::Serving::over`] exists
//! so that it does not have to.
//!
//! # What a frame is
//!
//! A roster of instances, a union page translation, one admission number, and
//! a list of steps in execution order. A step is a `LaunchPlan`: token ids,
//! per-request positions, the page CSR, and which rows read out. Everything
//! this driver needs to build a [`Request`] is in that plan, which is why the
//! conversion below is arithmetic and not a lookup.

use driver_api::{FrameSubmission, LaunchPlan};

use crate::resources::Request;
use crate::turns::{Step, Unstepped};

/// What a frame did.
///
/// The engine's three answers, and they are three because the caller's next
/// move differs completely: [`Self::Exhausted`] means try again after
/// evicting, [`Self::Impossible`] means never, and no amount of waiting
/// changes it.
#[derive(Debug)]
pub enum Launched {
    /// It ran. One [`Step`] per step of the frame, in execution order.
    Ran(Vec<Step>),
    /// The pool cannot hold this frame TODAY. Evict and re-post.
    Exhausted,
    /// The pool cannot hold this frame at any occupancy.
    ///
    /// Distinct from [`Self::Exhausted`] because a scheduler that waited on
    /// this would wait forever, and one that dropped an `Exhausted` frame
    /// would drop work it had correctly admitted.
    Impossible,
}

/// Why a frame could not be served at all.
#[derive(Debug)]
pub enum Unlaunched {
    /// A step's own CSRs do not describe a servable fire.
    ///
    /// Checked BEFORE anything is staged, which is the rule this crate keeps
    /// everywhere: decide, then move.
    Malformed(String),
    /// A step ran into the layer below.
    Unstepped(Unstepped),
    /// The frame names a verb this driver does not serve.
    Unserved(&'static str),
}

impl std::fmt::Display for Unlaunched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(why) => write!(f, "this frame is not servable: {why}"),
            Self::Unstepped(e) => write!(f, "a step of this frame did not run: {e:?}"),
            Self::Unserved(what) => write!(f, "this driver does not serve {what}"),
        }
    }
}

impl std::error::Error for Unlaunched {}

/// The pages a frame names, as a count the pool must cover.
///
/// The highest page NAMED and not the count required, because a frame's pages
/// are physical indices the scheduler picked anywhere in the pool: a frame
/// needing two pages can name page 900. `driver-metal`'s `launch` derives the
/// same number the same way, and for the same reason -- the trim task only
/// ever unmaps, so a pool left at a trimmed size refuses a page the scheduler
/// was right to hand out.
///
/// `u32::MAX` is the translation's hole and is skipped; a maximum over it
/// would size the pool for four billion pages.
#[must_use]
pub fn pages_named(frame: &FrameSubmission) -> u32 {
    frame
        .kv_translation
        .iter()
        .copied()
        .filter(|&p| p != u32::MAX)
        .max()
        .map_or(0, |p| p.saturating_add(1))
        .max(frame.required_kv_pages)
}

/// Every plan feature this driver does NOT implement, and whether the plan
/// asks for it.
///
/// # Why a refusal and not a silence
///
/// This is the class of bug the whole crate is built against. A driver that
/// ignores `max_layers` runs the full depth and answers fluently; one that
/// ignores `hook_page_mask` reads the pages the scheduler substituted AWAY
/// from and answers fluently; one that ignores `image_pixels` answers a
/// prompt whose picture was silently dropped, fluently. None of those is a
/// crash, a NaN or a validation error -- they are wrong text, and wrong text
/// is indistinguishable from right text without an oracle.
///
/// So each one is named. The engine may hand this driver a plan it cannot
/// serve, and the honest answer is which field, in the field's own name, at
/// admission -- before anything is written to a cache the scheduler would
/// then have to un-write.
///
/// `sampling_indices` is deliberately absent: [`crate::turns::Serving::over`]
/// forces every row to sample and says why there, so a sampling table is
/// OVERWRITTEN rather than ignored, and the rows the caller wanted are
/// recoverable from `Step::readout_of`.
///
/// `single_token_mode` is absent too: it is a hint about the shape of a fire
/// this driver already derives from `qo_indptr`, and honouring it is not a
/// behaviour, it is an optimisation.
#[must_use]
pub fn unserved_in(plan: &LaunchPlan) -> Option<&'static str> {
    // Order is the order the fields appear in `LaunchPlan`, so that a field
    // added there and not here is visible as a gap in this list rather than
    // as an answer nobody checked.
    if !plan.rs_slot_ids.is_empty() || !plan.rs_buffer_slot_ids.is_empty() {
        return Some("recurrent state: no model this driver serves holds any");
    }
    if plan.has_user_mask {
        return Some(
            "a user mask: attention here is causal, and a plan's mask \
                     would be dropped rather than applied",
        );
    }
    // A mask table with no user mask behind it is the engine's own synthesis,
    // and the only synthesis it makes is the CAUSAL mask -- `wire.rs` fills
    // `RunMask::all_true(pos + 1)` per row when it cannot elide the table.
    // Applying that and applying nothing are the same computation on a driver
    // whose attention is already causal, so refusing it refuses a plan this
    // driver serves exactly.
    //
    // It is not hypothetical and it is not rare: the elision is skipped
    // whenever a request is neither device-resolved nor a single-token
    // decode, which is every request in a batch that took the host-folded
    // path. Three concurrent `chat-completion`s through `pie serve` came back
    // as two failures and one success -- `this driver does not serve a user
    // mask`, about prompts that were entirely words.
    //
    // CHECKED, not assumed. `equals_causal` reads each mask's runs and its
    // length against the row's own position, so a mask that differs anywhere
    // -- a shorter window, a hole, a length that would admit the future --
    // still takes the refusal below.
    if !plan.masks.is_empty() && !masks_are_exactly_causal(plan) {
        return Some(
            "a mask that is not this driver's own causal one: attention \
                     here is causal, and a plan's mask would be dropped \
                     rather than applied",
        );
    }
    // A request that names SEVERAL readout rows, which is a shape this driver
    // cannot answer and used not to say so.
    //
    // `Serving::over` forces every row to sample and hands back one readout
    // PER TURN -- `Step::readout_of[i]` is turn `i`'s last row, the only one
    // that has seen the whole prompt. A caller that named `n` rows through
    // `fwd.readout(..)` gets one, and the program that reads them faults deep
    // inside the interpreter instead:
    //
    //     driver published poison epoch 1
    //     program faulted: launch program cannot be interpreted:
    //     logits intrinsic row range exceeds the forward's readout rows
    //
    // which is a real `pie serve` running `cacheback-speculative-decoding`.
    // The rows EXIST -- every row sampled -- and what is missing is a mapping
    // from a request to its row SPAN. Until there is one, this is a refusal at
    // the door rather than an answer with the wrong number of rows in it.
    if let Some(most) = widest_readout(plan) {
        if most > 1 {
            return Some(
                "a request naming several readout rows: this driver reads out \
                 one row per request, the last one",
            );
        }
    }
    if plan.max_layers.is_some() {
        return Some(
            "`max_layers`: a layer truncation this driver would run \
                     past, at full depth, fluently",
        );
    }
    if plan.hook_page_mask {
        return Some(
            "`hook_page_mask`: page substitution written by a hook \
                     stage, which this driver does not run",
        );
    }
    if plan.dense_device_mask {
        return Some(
            "`dense_device_mask`: a per-cell mask resolved from a \
                     channel, which this driver does not resolve",
        );
    }
    // The CONTENT, and the indptr's last boundary -- not whether the indptr
    // exists. These are CSRs with one boundary per request, so a batch of two
    // text-only requests carries `image_indptr = [0, 0, 0]`: present, and
    // naming nothing. Reading the vector's emptiness refused every multi-
    // request frame this driver was ever handed, with a message about images
    // that a chat completion had none of. A single request never showed it,
    // because the single-request path leaves the side-channels empty.
    if !plan.image_pixels.is_empty() || carries(&plan.image_indptr) {
        return Some("images: this driver serves text-only models");
    }
    if !plan.audio_features.is_empty() || carries(&plan.audio_indptr) {
        return Some("audio: this driver serves text-only models");
    }
    if !plan.embed_rows.is_empty() || carries(&plan.embed_indptr) {
        return Some("pre-embedded rows: this driver embeds from token ids");
    }
    None
}

/// Is every mask in this plan exactly the causal mask this driver applies?
///
/// # The structure, which is not the obvious one
///
/// `mask_indptr` is PER REQUEST, not per row -- `wire::emit_attention_masks`
/// pushes one boundary after appending a request's rows -- so the table has
/// `R + 1` entries for `R` requests and holds one BRLE per QUERY ROW inside
/// each span. Reading it as per-row costs nothing on a batch of one-row
/// decodes, where rows and requests coincide, and refuses every prefill.
///
/// # What "exactly" means
///
/// Row `j` of a request with `n` query rows over `kv_len` keys attends,
/// causally, to `kv_len - n + j + 1` of them: the last row sees the whole
/// span, and each earlier row one fewer. So the mask this driver already
/// computes is `all_true` of exactly that length, which BRLE spells `[0, m]`.
/// Both halves are checked:
///
/// * The runs must be true everywhere -- see [`all_true`], which reads any
///   encoding of that rather than only the canonical `[0, m]`.
/// * The length must be that row's own extent. A SHORTER all-true mask is a
///   sliding window this driver would widen; a LONGER one would admit keys
///   past the row, which the mask permits and causal attention does not.
///
/// `kv_len` is the POST-TRIM span, which is what makes this work at all: when
/// `TrimPlan` drops KV ranges it rewrites the BRLEs to match, and both sides
/// of this comparison move together because both are read off the same plan.
///
/// A plan whose tables do not describe each other answers false. An unreadable
/// mask table is not a causal one.
fn masks_are_exactly_causal(plan: &LaunchPlan) -> bool {
    // One boundary per request plus one, and one `kv_len` per request.
    let requests = plan.kv_len.len();
    if requests == 0 || plan.mask_indptr.len() != requests + 1 {
        return false;
    }
    if plan.qo_indptr.len() != requests + 1 {
        return false;
    }
    if plan.mask_indptr.last().copied().unwrap_or(0) as usize != plan.masks.len() {
        return false;
    }
    for r in 0..requests {
        let (mask_from, mask_to) = (
            plan.mask_indptr[r] as usize,
            plan.mask_indptr[r + 1] as usize,
        );
        // A request that names no mask is the elided case: nothing to drop.
        if mask_from == mask_to {
            continue;
        }
        if mask_to < mask_from || mask_to > plan.masks.len() {
            return false;
        }
        let rows = (plan.qo_indptr[r + 1] as usize).checked_sub(plan.qo_indptr[r] as usize);
        let Some(rows) = rows else {
            return false;
        };
        // One BRLE per query row, or this is a shape with no reading.
        if mask_to - mask_from != rows {
            return false;
        }
        let kv_len = u64::from(plan.kv_len[r]);
        for (j, mask) in plan.masks[mask_from..mask_to].iter().enumerate() {
            // `kv_len - n + j + 1`, in an order that cannot underflow.
            let Some(extent) = (kv_len + j as u64 + 1).checked_sub(rows as u64) else {
                return false;
            };
            if mask.total_size != extent || !all_true(mask) {
                return false;
            }
        }
    }
    true
}

/// The most readout rows any one request in this plan names.
///
/// Public because two paths have to ask it: `unserved_in` for a plan that
/// arrives whole, and `envelope::fill` for one whose table it is about to
/// drop. One function so the two cannot drift into different readings of the
/// same CSR.
///
/// `None` for a plan with no sampling table, which is the ordinary case: the
/// engine elides it when every request wants its last row, and this driver
/// gives exactly that.
pub fn widest_readout(plan: &LaunchPlan) -> Option<u32> {
    if plan.sampling_indptr.len() < 2 {
        return None;
    }
    plan.sampling_indptr
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]))
        .max()
}

/// Is this BRLE true everywhere, however it is encoded?
///
/// BRLE alternates starting with FALSE, so "all true" is any run list whose
/// even-index runs are zero and whose odd ones cover the whole length. The
/// canonical form is `[0, n]`, but `TrimPlan::write_skipping` rebuilds a
/// mask's runs when it drops KV ranges and can leave zero-length falses
/// between the pieces it kept -- which is exactly the case this function is
/// reached for, so requiring the canonical pair would refuse the mask that
/// motivated reading it at all.
fn all_true(mask: &driver_api::plan::EncodedMask) -> bool {
    let mut at = 0u64;
    for (index, &run) in mask.runs.iter().enumerate() {
        if at >= mask.total_size {
            break;
        }
        if index % 2 == 0 && run > 0 {
            return false;
        }
        at = at.saturating_add(u64::from(run));
    }
    at >= mask.total_size
}

/// Does this CSR name any rows at all?
///
/// Its LAST boundary is the total, so a table of any length whose total is
/// zero holds nothing. An empty table holds nothing either.
fn carries(indptr: &[u32]) -> bool {
    indptr.last().is_some_and(|&total| total > 0)
}

/// A program pass that faulted: the instance, and what it said.
pub type Fault = (u64, String);

/// Fire every program this step's roster names, each over its own rows of the
/// step's distribution.
///
/// # Why this exists
///
/// The channel plane was registration-only. A frame could bind an instance,
/// the driver would launch the model rows, and the program -- the sampler --
/// never ran, so the engine read an unadvanced ring and the fire's answer
/// went nowhere. `crate::programs`' module doc used to end by saying exactly
/// that. This is the verb that closes it.
///
/// # Why it is here and not in [`crate::shell::Shell`]
///
/// The registry is not the shell's. It is alive from the driver's `create`
/// rather than from its `load_model`, because nothing in the channel plane is
/// about a model -- see [`crate::programs`]. So the loop lives here, where it
/// is pure host code a test can run with no device at all, and the caller
/// holds the two halves.
///
/// # Rows, and the mistake this avoids
///
/// [`crate::turns::Serving::over`] forces every ROW to sample, so
/// `step.logits` holds one distribution per token in FIRE order, and
/// `step.readout_of[r]` says which of those rows is request `r`'s answer.
/// The interpreter reads its inputs from `base_row = 0` and cannot be told
/// otherwise, so a member's rows are GATHERED into a fresh buffer rather than
/// passed as a range into the whole read-out. Passing the whole read-out is
/// the defect `driver-metal` shipped and fixed: every member sampled the
/// FIRST request's distribution and returned its token -- one fire, N
/// requests, one answer repeated, nothing faulted, and invisible to any
/// single-request test.
///
/// # Why a fault is not an error
///
/// It poisons the ONE instance that faulted. Failing the frame would take
/// down every other request batched with it for a fault that is one
/// program's, and those requests ran. Faults are appended for the caller to
/// report. A blocked pass is not an error either, for a different reason:
/// readiness is the program's own gate, and missing it means the pass did not
/// happen and changed nothing, so the caller re-posts.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] for a roster row naming no instance of this
/// frame -- a frame built against a registry the scheduler did not have --
/// for a request whose read-out row does not exist, and for an instance the
/// registry does not hold.
pub fn run_programs(
    programs: &mut crate::programs::Programs,
    instance_ids: &[u64],
    sub: &driver_api::StepSubmission,
    step: &Step,
    faults: &mut Vec<Fault>,
) -> Result<Vec<Ran>, Unlaunched> {
    let vocab = step.logits.vocab;
    let mut ran = Vec::with_capacity(sub.roster_rows.len());
    for (member, &row) in sub.roster_rows.iter().enumerate() {
        let id = *instance_ids.get(row as usize).ok_or_else(|| {
            Unlaunched::Malformed(format!(
                "roster row {row} is outside this frame's {} instances",
                instance_ids.len()
            ))
        })?;
        // Composed-against, or composed against a ring that has since moved.
        // A ticket is the only check that can tell those apart: see
        // [`tickets_for`].
        match tickets_for(sub, member).map_or(Ok(driver::Readiness::Ready), |tickets| {
            programs
                .ready(id, &tickets)
                .map_err(|e| Unlaunched::Malformed(format!("{e}")))
        })? {
            driver::Readiness::Ready => {}
            // Early, and not wrong. The producer has not run, the consumer has
            // not drained, or another fire moved the ring: the member is
            // skipped, nothing about it changes, and the scheduler re-posts.
            driver::Readiness::Retry { .. } => {
                ran.push(Ran::Early);
                continue;
            }
            // Never runnable: a poisoned or closed ring. One instance's
            // problem, so it is a fault beside the others rather than the
            // whole frame's failure -- and a fault a waiter can see, since
            // the ring already carries the word the pipeline reads.
            driver::Readiness::Failed { channel, reason } => {
                faults.push((id, format!("channel {channel:?}: {reason:?}")));
                ran.push(Ran::Faulted);
                continue;
            }
            // The frame's tables do not describe the instance they name.
            other => {
                return Err(Unlaunched::Malformed(format!(
                    "instance {id} was posted with tickets that do not fit it: {other:?}"
                )));
            }
        }
        let (lo, hi) = member_requests(&sub.program_row_indptr, member, step.readout_of.len());
        let mut values = Vec::with_capacity((hi - lo) * vocab);
        for r in lo..hi {
            let at = step.readout_of[r];
            let one = step.logits.row(at).ok_or_else(|| {
                Unlaunched::Malformed(format!(
                    "request {r} reads row {at} of a read-out of {} rows",
                    step.logits.rows
                ))
            })?;
            values.extend_from_slice(one);
        }
        // An empty span is the device-geometry placeholder -- a member that
        // owns no read-out row -- and it fires with NO forward rather than
        // with row zero's, which is the same distinction `PassInputs::none`
        // exists to make.
        let inputs = if values.is_empty() {
            driver::PassInputs::none()
        } else {
            driver::PassInputs {
                logits: Some(&values),
                rows: (hi - lo) as u32,
                vocab: vocab as u32,
                mtp_draft_row: None,
            }
        };
        match programs.fire(id, &inputs) {
            Ok(driver::StepOutcome::Committed | driver::StepOutcome::Blocked(_)) => {
                ran.push(Ran::Fired);
            }
            Ok(driver::StepOutcome::Faulted(why)) => {
                faults.push((id, why));
                ran.push(Ran::Faulted);
            }
            Err(e) => return Err(Unlaunched::Malformed(format!("{e}"))),
        }
    }
    Ok(ran)
}

/// What one member of a step did, in roster order.
///
/// # Why the caller is told, rather than only the faults
///
/// Every member of a launched frame owns a TERMINAL CELL, and the engine
/// resolves the request's work item by reading it: `Pending` is not "nothing
/// happened", it is a failure -- `work item completion terminal outcome is
/// still Pending` -- and a member that was skipped for being early has to say
/// `Retry` or the scheduler turns a re-postable frame into a dead request.
///
/// So a fault list is not enough. It names the members that failed and cannot
/// distinguish the two REMAINING outcomes from each other, and those two are
/// the ones whose cells differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ran {
    /// The member's program fired, committed or blocked on its own await.
    Fired,
    /// The member was not ready: its producer has not run or its ring has
    /// moved. Nothing about it changed and the frame may be re-posted.
    Early,
    /// The member's program or one of its channels failed.
    Faulted,
}

/// The head and tail this member's composer pinned, or `None` for a member it
/// pinned nothing for.
///
/// # Why the absent case is `None` and not a defaulted table
///
/// `channel_ticket_indptr` partitions `channel_expected_head` and
/// `channel_expected_tail` by member, and a frame that pins nothing states
/// none of the three. Filling that in with unpinned entries looks like the
/// permissive answer and is the opposite: [`driver::check`] answers
/// `Reason::Unpinned` -- a RETRY -- for any fire that puts without a pinned
/// tail, so a defaulted table would park every putting program forever, which
/// reads as a hang rather than as a refusal.
///
/// `None` therefore means "the composer made no decision here, so there is
/// nothing to check that it still holds", and the member fires on the
/// interpreter's own readiness gate alone -- which is what this driver did
/// before tickets were honoured at all, and what `driver-metal`'s host path
/// still does.
fn tickets_for(sub: &driver_api::StepSubmission, member: usize) -> Option<Vec<driver::Ticket>> {
    let (lo, hi) = match (
        sub.channel_ticket_indptr.get(member),
        sub.channel_ticket_indptr.get(member + 1),
    ) {
        (Some(&s), Some(&e)) if e > s => (s as usize, e as usize),
        _ => return None,
    };
    Some(
        (lo..hi)
            .map(|t| driver::Ticket {
                expected_head: sub
                    .channel_expected_head
                    .get(t)
                    .copied()
                    .unwrap_or(driver::NO_TICKET),
                expected_tail: sub
                    .channel_expected_tail
                    .get(t)
                    .copied()
                    .unwrap_or(driver::NO_TICKET),
            })
            .collect(),
    )
}

/// Which of a step's requests belong to batch member `member`.
///
/// # Why a member is not a request
///
/// A frame's roster is one entry per PROGRAM instance and a step's fire is
/// one distribution per REQUEST, and the two are not the same list: a
/// speculative member owns several read-out rows, and a member whose geometry
/// the device resolves owns an empty placeholder. `program_row_indptr` is the
/// frame's own attribution CSR and says which is which -- member `p` owns wire
/// request rows `[indptr[p], indptr[p + 1])`.
///
/// This is worth a named function and a test because getting it wrong is
/// invisible. `driver-metal` handed every member the WHOLE read-out, and the
/// interpreter's `base_row` is 0, so in a frame of three requests all three
/// programs sampled the FIRST request's distribution and returned its token.
/// One fire, three answers, all the same, nothing faulted -- and no
/// single-request test can see it.
///
/// An absent or unusable CSR gives the whole read-out to the member, which is
/// the single-member case and the behaviour `driver-metal` falls back to for
/// the same shape.
#[must_use]
pub fn member_requests(
    program_row_indptr: &[u32],
    member: usize,
    requests: usize,
) -> (usize, usize) {
    match (
        program_row_indptr.get(member),
        program_row_indptr.get(member + 1),
    ) {
        (Some(&s), Some(&e)) if e >= s && e as usize <= requests => (s as usize, e as usize),
        _ => (0, requests),
    }
}

/// One step's requests, in the plan's own request order.
///
/// # The conversion
///
/// A request is `qo_indptr[r]..qo_indptr[r + 1]` rows of the batch, its pages
/// are `kv_page_indices[kv_page_indptr[r]..kv_page_indptr[r + 1]]`, and its
/// positions are the plan's `position_ids` for those rows -- which are
/// per-request already, which is what makes the page arithmetic in
/// [`Request`] a division.
///
/// # Why `sampling_indices` is dropped
///
/// [`crate::turns::Serving::over`] forces every row to sample, for the arena
/// reason recorded there, and rewrites the sampling table to the identity to
/// match. A `samples` here would be overwritten one line later, so stating it
/// would be a lie in the record rather than a value anyone reads.
///
/// This used to add *"the rows a caller wanted are recoverable from
/// `Step::readout_of` plus the plan"*, and that was FALSE for the case it
/// mattered in. `readout_of` is one entry PER TURN -- a turn's answer is its
/// LAST row -- so a request that named several readout rows gets one back, and
/// there is nothing to recover the others from. `unserved_in` refuses such a
/// plan by name now, which is what the claim should have been.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] naming the single CSR that does not close.
pub fn requests_of(plan: &LaunchPlan) -> Result<Vec<Request>, Unlaunched> {
    let rows = plan.qo_indptr.len().saturating_sub(1);
    if rows == 0 {
        return Err(Unlaunched::Malformed(
            "qo_indptr has no requests in it".to_string(),
        ));
    }
    if plan.kv_page_indptr.len() != rows + 1 {
        return Err(Unlaunched::Malformed(format!(
            "qo_indptr describes {rows} requests and kv_page_indptr describes {}",
            plan.kv_page_indptr.len().saturating_sub(1)
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let (lo, hi) = (plan.qo_indptr[r] as usize, plan.qo_indptr[r + 1] as usize);
        // Ascending and in range, checked rather than assumed: a descending
        // pair slices backwards and panics, and a `hi` past the end takes
        // whatever follows the vector's own data as positions.
        if lo > hi || hi > plan.position_ids.len() {
            return Err(Unlaunched::Malformed(format!(
                "request {r} spans rows {lo}..{hi} of {} positions",
                plan.position_ids.len()
            )));
        }
        let (plo, phi) = (
            plan.kv_page_indptr[r] as usize,
            plan.kv_page_indptr[r + 1] as usize,
        );
        if plo > phi || phi > plan.kv_page_indices.len() {
            return Err(Unlaunched::Malformed(format!(
                "request {r} spans pages {plo}..{phi} of {} page indices",
                plan.kv_page_indices.len()
            )));
        }
        if lo == hi {
            // A request contributing no rows is not expressible below --
            // `Request::rows` would be empty and `Frame::of` would seriate
            // nothing for it -- and silently dropping it would renumber every
            // later request's readout. Refused by name instead.
            return Err(Unlaunched::Malformed(format!(
                "request {r} contributes no rows, so its readout has no answer to be"
            )));
        }
        out.push(Request::of(
            plan.position_ids[lo..hi].to_vec(),
            plan.kv_page_indices[plo..phi].to_vec(),
        ));
    }
    Ok(out)
}

/// One step's tokens, split per request the same way.
///
/// Separate from [`requests_of`] because a request states WHERE its rows go
/// and a token states what is in one; the split is by the same CSR, and
/// keeping them apart is what let the placement be tested without a page.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] if the token vector is shorter than the row
/// span, which would otherwise feed a conversation whatever followed it.
pub fn tokens_of(plan: &LaunchPlan) -> Result<Vec<Vec<u32>>, Unlaunched> {
    let rows = plan.qo_indptr.len().saturating_sub(1);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let (lo, hi) = (plan.qo_indptr[r] as usize, plan.qo_indptr[r + 1] as usize);
        if lo > hi || hi > plan.token_ids.len() {
            return Err(Unlaunched::Malformed(format!(
                "request {r} spans rows {lo}..{hi} of {} token ids",
                plan.token_ids.len()
            )));
        }
        out.push(plan.token_ids[lo..hi].to_vec());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(qo: &[u32], toks: &[u32], pos: &[u32], pidx: &[u32], pptr: &[u32]) -> LaunchPlan {
        LaunchPlan {
            token_ids: toks.to_vec(),
            position_ids: pos.to_vec(),
            kv_page_indices: pidx.to_vec(),
            kv_page_indptr: pptr.to_vec(),
            qo_indptr: qo.to_vec(),
            ..LaunchPlan::default()
        }
    }

    /// Two requests of different lengths are split at the CSR's boundaries.
    ///
    /// # Why this is the test that matters
    ///
    /// Every field here is a slice of a shared vector, and an off-by-one in
    /// any of the four bounds produces a servable fire that answers the wrong
    /// conversation: request 1 reading request 0's last position appends its
    /// key over a token that already exists, and the model stays fluent.
    #[test]
    fn a_frame_s_csr_splits_into_the_requests_it_describes() {
        // Request 0: rows 0..3 (a prefill of three), pages 4 and 9.
        // Request 1: row 3 (a decode), page 2.
        let p = plan(
            &[0, 3, 4],
            &[100, 101, 102, 200],
            &[0, 1, 2, 7],
            &[4, 9, 2],
            &[0, 2, 3],
        );
        let requests = requests_of(&p).expect("a well-formed plan");
        let tokens = tokens_of(&p).expect("a well-formed plan");

        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].positions,
            vec![0, 1, 2],
            "request 0's positions are not its own three rows"
        );
        assert_eq!(
            requests[0].pages,
            vec![4, 9],
            "request 0 was given pages that are not the ones its CSR names"
        );
        assert_eq!(
            requests[1].positions,
            vec![7],
            "request 1's decode is at position 7, and a slice from the wrong \
             end of the CSR gives it request 0's"
        );
        assert_eq!(requests[1].pages, vec![2]);
        assert_eq!(tokens, vec![vec![100, 101, 102], vec![200]]);
    }

    /// A CSR that does not close is refused, rather than sliced.
    #[test]
    fn a_csr_that_runs_past_its_own_data_is_refused() {
        // `qo_indptr` claims four rows and there are three positions.
        let p = plan(&[0, 4], &[1, 2, 3], &[0, 1, 2], &[0], &[0, 1]);
        requests_of(&p).expect_err("four rows of three positions");

        // A page span past the page indices.
        let p = plan(&[0, 1], &[1], &[0], &[3], &[0, 5]);
        requests_of(&p).expect_err("five pages out of one");

        // Two CSRs describing different numbers of requests.
        let p = plan(&[0, 1, 2], &[1, 2], &[0, 0], &[3], &[0, 1]);
        requests_of(&p).expect_err("two requests and one page span");

        // A request that contributes no rows.
        let p = plan(&[0, 0, 1], &[1], &[0], &[3, 4], &[0, 1, 2]);
        requests_of(&p).expect_err("request 0 has no rows");
    }

    /// The pool must cover the highest page a frame NAMES, not its count.
    ///
    /// A frame needing two pages can name page 900, because the scheduler
    /// picks physical indices anywhere in the pool. Sizing to the count sends
    /// a fire at a page the pool does not have.
    #[test]
    fn a_frame_is_sized_by_the_pages_it_names() {
        let frame = FrameSubmission {
            instance_ids: vec![1],
            kv_translation: vec![900, u32::MAX, 12],
            kv_translation_indptr: vec![0, 3],
            required_kv_pages: 2,
            steps: Vec::new(),
        };
        assert_eq!(
            pages_named(&frame),
            901,
            "the pool was sized from the page COUNT, so the fire addresses a \
             page the pool does not have"
        );

        // ...and the hole is skipped rather than maximised over.
        let frame = FrameSubmission {
            kv_translation: vec![u32::MAX],
            kv_translation_indptr: vec![0, 1],
            required_kv_pages: 3,
            ..frame
        };
        assert_eq!(
            pages_named(&frame),
            3,
            "u32::MAX is the translation's hole, and a pool sized from it \
             would want sixteen terabytes a layer"
        );
    }

    /// Every field `unserved_in` names actually refuses, and a plain plan
    /// does not.
    ///
    /// # Why a table and not eight tests
    ///
    /// Because the property is the PARTITION: what this driver ignores must
    /// be exactly what it refuses. A field added to `LaunchPlan` and served
    /// nowhere is invisible to any test that names fields one at a time --
    /// this at least keeps the ones already known in one place, next to the
    /// list they are checked against.
    ///
    /// The baseline assertion is the load-bearing half. A `unserved_in` that
    /// refused everything would pass every case below except that one, and a
    /// driver that refused every frame is a driver that serves nothing.
    #[test]
    fn a_plan_naming_what_this_driver_cannot_do_is_refused_by_that_name() {
        let base = plan(&[0, 1], &[100], &[0], &[4], &[0, 1]);
        assert!(
            unserved_in(&base).is_none(),
            "a plain text decode is refused, so this driver serves nothing"
        );

        let cases: Vec<(&str, LaunchPlan)> = vec![
            (
                "recurrent state",
                LaunchPlan {
                    rs_slot_ids: vec![0],
                    ..base.clone()
                },
            ),
            (
                "a user mask",
                LaunchPlan {
                    has_user_mask: true,
                    ..base.clone()
                },
            ),
            (
                "`max_layers`",
                LaunchPlan {
                    max_layers: Some(4),
                    ..base.clone()
                },
            ),
            (
                "`hook_page_mask`",
                LaunchPlan {
                    hook_page_mask: true,
                    ..base.clone()
                },
            ),
            (
                "`dense_device_mask`",
                LaunchPlan {
                    dense_device_mask: true,
                    ..base.clone()
                },
            ),
            (
                "images",
                LaunchPlan {
                    image_pixels: vec![7],
                    ..base.clone()
                },
            ),
            (
                "audio",
                LaunchPlan {
                    audio_features: vec![7],
                    ..base.clone()
                },
            ),
            (
                "pre-embedded rows",
                LaunchPlan {
                    embed_rows: vec![7],
                    ..base.clone()
                },
            ),
        ];

        for (name, p) in &cases {
            let why =
                unserved_in(p).unwrap_or_else(|| panic!("{name} is ignored rather than refused"));
            assert!(
                why.contains(name),
                "{name} is refused, but the refusal says {why:?} instead, which \
                 leaves the caller to guess which field it was"
            );
        }

        // Every side-channel names ONE boundary PER REQUEST, so a batch of two
        // text-only requests carries three zeros in each of them. That is the
        // shape of a table with nothing in it, and it was refused as images
        // for as long as this driver only ever saw one request at a time --
        // the first real two-conversation turn came back as `this driver does
        // not serve images`, about a prompt that was entirely words.
        let two = LaunchPlan {
            image_indptr: vec![0, 0, 0],
            audio_indptr: vec![0, 0, 0],
            embed_indptr: vec![0, 0, 0],
            ..base.clone()
        };
        assert!(
            unserved_in(&two).is_none(),
            "a text-only batch of two is refused for a side-channel that holds nothing"
        );
        // And the boundary is still read: a table whose total is non-zero
        // names rows, whatever its length.
        assert_eq!(
            unserved_in(&LaunchPlan {
                image_indptr: vec![0, 1],
                ..base.clone()
            }),
            Some("images: this driver serves text-only models"),
            "an image the plan names only in its CSR is served silently"
        );
    }

    /// The engine's own causal mask is served; anything else is refused.
    ///
    /// # The plan this was measured on
    ///
    /// `wire.rs` synthesizes `RunMask::all_true` per query row whenever it
    /// cannot elide the mask table, which is every request that is neither
    /// device-resolved nor a single-token decode. Three concurrent
    /// `chat-completion`s through a real `pie serve` came back as one success
    /// and two failures -- *this driver does not serve a user mask* -- about
    /// prompts that were entirely words. The mask was the causal one, which is
    /// the attention this driver already computes, so ignoring it and applying
    /// it are the same bytes.
    ///
    /// # Why each rejection below is a separate case
    ///
    /// Accepting "all true" alone would be wrong, and the cases say how. A
    /// mask SHORTER than the row's extent is a sliding window; one LONGER
    /// admits keys past the row, which causal attention refuses and the mask
    /// does not; a run list that is not `[0, m]` has a hole in it. In each the
    /// two computations differ, so none is served by dropping the table.
    ///
    /// The multi-ROW request is the case that matters most, because
    /// `mask_indptr` is per REQUEST: a first draft read it as per-row, passed
    /// every one-row decode, and refused every prefill. The six-way concurrent
    /// run that caught that is the reason this test has a prefill in it.
    #[test]
    fn the_engines_own_causal_mask_is_served_and_no_other_is() {
        let brle = |m: u32| driver_api::EncodedMask::new(vec![0, m], u64::from(m));

        // Two requests: a three-row prefill over a 3-key span, and a one-row
        // decode over a 5-key span. Two rows and two requests would not tell
        // a per-row reading of `mask_indptr` from a per-request one.
        let base = LaunchPlan {
            qo_indptr: vec![0, 3, 4],
            token_ids: vec![10, 11, 12, 13],
            position_ids: vec![0, 1, 2, 4],
            kv_len: vec![3, 5],
            kv_page_indices: vec![0, 1],
            kv_page_indptr: vec![0, 1, 2],
            ..LaunchPlan::default()
        };
        assert!(
            unserved_in(&base).is_none(),
            "the same plan without a mask table is served, so what follows is \
             about the table and nothing else"
        );

        // What `wire.rs` builds: row j of an n-row request over kv_len keys
        // attends to `kv_len - n + j + 1` of them.
        let served = LaunchPlan {
            masks: vec![brle(1), brle(2), brle(3), brle(5)],
            mask_indptr: vec![0, 3, 4],
            ..base.clone()
        };
        assert_eq!(
            unserved_in(&served),
            None,
            "the causal mask is the attention this driver computes"
        );

        // A request that names NO mask is the elided case and mixes with one
        // that does.
        assert_eq!(
            unserved_in(&LaunchPlan {
                masks: vec![brle(5)],
                mask_indptr: vec![0, 0, 1],
                ..base.clone()
            }),
            None
        );

        let refused: Vec<(&str, Vec<driver_api::EncodedMask>, Vec<u32>)> = vec![
            (
                "a window, not the row's full extent",
                vec![brle(1), brle(2), brle(2), brle(5)],
                vec![0, 3, 4],
            ),
            (
                "longer than the row's extent",
                vec![brle(1), brle(2), brle(3), brle(9)],
                vec![0, 3, 4],
            ),
            (
                "a hole in the runs",
                vec![
                    brle(1),
                    brle(2),
                    driver_api::EncodedMask::new(vec![1, 2], 3),
                    brle(5),
                ],
                vec![0, 3, 4],
            ),
            (
                "the prefill read as one row",
                vec![brle(3), brle(5)],
                vec![0, 1, 2],
            ),
            (
                "a table that does not close",
                vec![brle(1), brle(2), brle(3), brle(5)],
                vec![0, 3, 5],
            ),
        ];
        for (what, masks, mask_indptr) in refused {
            let plan = LaunchPlan {
                masks,
                mask_indptr,
                ..base.clone()
            };
            let said =
                unserved_in(&plan).unwrap_or_else(|| panic!("{what} must not be served silently"));
            assert!(
                said.contains("causal"),
                "{what}: the refusal should say what it wanted: {said}"
            );
        }

        // A NON-CANONICAL all-true encoding is served. `write_skipping`
        // rebuilds a mask's runs when the trim drops KV ranges and can leave
        // zero-length falses between the pieces it kept; a check that demanded
        // `[0, m]` would refuse the very mask trimming produces.
        assert_eq!(
            unserved_in(&LaunchPlan {
                masks: vec![
                    brle(1),
                    brle(2),
                    driver_api::EncodedMask::new(vec![0, 1, 0, 2], 3),
                    driver_api::EncodedMask::new(vec![0, 2, 0, 3], 5),
                ],
                mask_indptr: vec![0, 3, 4],
                ..base.clone()
            }),
            None,
            "all-true is a property of the mask, not of how it was written down"
        );

        // The length is read off `kv_len`, which is the POST-TRIM span, and
        // that is where this parts from `driver-vulkan`'s `causal_only`: it
        // compares against `position_ids[query] + 1`, which is ABSOLUTE. The
        // two agree exactly when nothing was trimmed -- for an n-row request
        // at base position p, `kv_len - n + j + 1` IS `p + j + 1` -- and part
        // when a trim has shortened the span the driver actually binds. Here
        // the positions say 8 and the bound span says 5; the mask the engine
        // wrote matches the span.
        assert_eq!(
            unserved_in(&LaunchPlan {
                position_ids: vec![6, 7, 8, 12],
                masks: vec![brle(1), brle(2), brle(3), brle(5)],
                mask_indptr: vec![0, 3, 4],
                ..base.clone()
            }),
            None,
            "a trimmed span is measured against what the plan binds, not \
             against where the tokens are"
        );

        // And a GUEST mask is refused whatever its shape, because
        // `has_user_mask` says a caller asked for something.
        assert!(
            unserved_in(&LaunchPlan {
                has_user_mask: true,
                masks: vec![brle(1), brle(2), brle(3), brle(5)],
                mask_indptr: vec![0, 3, 4],
                ..base
            })
            .is_some_and(|s| s.contains("user mask")),
            "a mask the guest asked for is refused even when it is causal"
        );
    }

    /// One readout row per request is served; several is refused by name.
    ///
    /// # Why the refusal is the right answer and not a smaller one
    ///
    /// `Serving::over` forces every row to sample, so the rows a caller asked
    /// for EXIST in `Step::logits`. What does not exist is a mapping from a
    /// request to its row SPAN: `readout_of` is one entry per turn, the last
    /// row, the only one that has seen the whole prompt. Handing a program one
    /// row where it asked for four is not a degraded answer, it is a different
    /// one -- and until this refusal existed the program found out by faulting
    /// inside `crates/driver`'s interpreter, four layers from the plan that
    /// caused it:
    ///
    ///     logits intrinsic row range exceeds the forward's readout rows
    ///
    /// The ordinary case must keep working, which is the other half of this
    /// test: the engine ELIDES the sampling table when every request wants its
    /// last row, and a plan with no table is what twenty inferlets submit.
    #[test]
    fn a_request_naming_several_readout_rows_is_refused_by_name() {
        let base = plan(&[0, 1, 3], &[10, 11, 12], &[0, 1, 2], &[0, 1], &[0, 1, 2]);
        assert!(
            unserved_in(&base).is_none(),
            "a plan with no sampling table is the ordinary case and is served"
        );
        // One row per request, stated: still the ordinary case.
        assert_eq!(
            unserved_in(&LaunchPlan {
                sampling_indices: vec![0, 2],
                sampling_indptr: vec![0, 1, 2],
                ..base.clone()
            }),
            None,
            "one readout row per request is exactly what this driver gives"
        );
        // The second request wants two. That is the speculative verifier's
        // shape, and it is refused.
        let refused = unserved_in(&LaunchPlan {
            sampling_indices: vec![0, 1, 2],
            sampling_indptr: vec![0, 1, 3],
            ..base.clone()
        })
        .expect("a request naming two readout rows is not served silently");
        assert!(
            refused.contains("readout"),
            "the refusal names what it could not do: {refused}"
        );
        // And a table that names NO rows for anybody is not a refusal: nothing
        // was asked for.
        assert_eq!(
            unserved_in(&LaunchPlan {
                sampling_indices: Vec::new(),
                sampling_indptr: vec![0, 0, 0],
                ..base
            }),
            None,
            "an empty sampling table asks for nothing, so there is nothing to refuse"
        );
    }
}
