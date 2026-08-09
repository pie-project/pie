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
///
/// # What a refusal looks like from the far end
///
/// Measured rather than assumed, by running `sliding-window-attention` --
/// which is a real inferlet in this tree, not a fixture -- against a real
/// `pie` on this backend:
///
/// ```text
/// first_token take: channel is poisoned: pipeline: forward failed:
/// direct launch rejected: driver-vulkan: this driver does not serve a
/// user mask: attention here is causal, and a plan's mask would be
/// dropped rather than applied
/// ```
///
/// That is the whole point of naming the field. The guest is told which
/// capability is missing, in the sentence written here, at the first launch
/// -- not a page of tokens later, and not as fluent text that is quietly
/// attending outside its window.
///
/// # That refusal was not honest, and three inferlets paid for it
///
/// This section used to end by saying the refusal was honest and the gap was
/// elsewhere. It was not. `has_user_mask` was the FIRST half of the condition
/// below, so a guest that named a mask had its rows refused without them ever
/// being read -- and then `causal_row` read a row's WIDTH where it meant to
/// read its cells, so even had the flag not short-circuited, a rectangle
/// would have been refused for being a rectangle.
///
/// A guest builds a mask as `[queries, pool_len]`, because that is the shape a
/// pool has. Row 0 of one arrives here as `runs=[0, 1, 47]` over
/// `total_size=48`: one true cell and forty-seven false ones. That is causal.
/// It was measured coming off the wire under an `eprintln!` in `causal_only`,
/// from `contrastive-decoding` with its window opened past the pool, where
/// the guest's own mask reduces to `key <= query` exactly.
///
/// What that cost, measured on this backend before and after:
///
/// * `contrastive-decoding` with a wide window was refused; it now runs and
///   answers. With its default window of 8 it is refused, correctly, because
///   that mask is a real restriction.
/// * `sliding-window-attention` and `attention-sink` were refused at their
///   PREFILL, whose mask is `key <= query` and nothing else. They now get
///   past this driver entirely and stop at an engine wall instead --
///   `EmbedTokens is not host-derivable: channel 2 has no host-known value`,
///   reached after the engine reports that `a channel-bound dense AttnMask
///   belongs to the pool-owned device-geometry class` and falls back to
///   host-evaluated execution. `beam-search` stops at the same wall. That is
///   not this driver's refusal and not this driver's to lift.
///
/// The refusal that remains IS honest, and it is narrower than the one that
/// was here: a mask whose true cells are not exactly `0..=position`.
///
/// # Where the missing piece actually is
///
/// Not in the kernels, which was the assumption worth checking.
/// `attn/sdpa_paged.comp` already declares all three of the things these
/// inferlets want: a dense per-row mask at bindings 8 and 9 with an
/// `attention_mask_stride`, a `window` that starts the key scan at
/// `q_pos - window + 1`, and `PIE_WITH_SINK` variants that merge a per-head
/// learned logit into the softmax -- `sdpa_paged_decode_sink_bfloat16_d_64`
/// and three siblings are in the built entrypoint list. `sdpa_sliding.comp`
/// adds `sdpa_vector_decode_swa_bfloat16_d_{256,512}`, and those two widths
/// are the whole of it: there is no `_swa_d_128`, which is the width the
/// checkpoint served here uses.
///
/// The gap is one layer up. `model_compiler::dsl`'s `sdpa` states its
/// operands as the query and the KV state and nothing else, and passes a mask
/// stride of a literal `0` -- so no rectangle NAMES a mask, and a driver
/// cannot bind a buffer no operand asks for. The `window` is already a
/// parameter of that statement, so a model whose TEXT states a window is
/// served today; what is not served is a mask that varies per REQUEST.
///
/// So the work is a row change in shared lowering code, which every backend's
/// binding audit is pinned against, and it is deliberately not done from
/// here. What this crate can say is that the kernels are ready for the day
/// the row names them, and that everything short of a real per-request
/// restriction is served rather than refused.
#[must_use]
pub fn unserved_in(plan: &LaunchPlan) -> Option<&'static str> {
    // Order is the order the fields appear in `LaunchPlan`, so that a field
    // added there and not here is visible as a gap in this list rather than
    // as an answer nobody checked.
    if !plan.rs_slot_ids.is_empty() || !plan.rs_buffer_slot_ids.is_empty() {
        return Some("recurrent state: no model this driver serves holds any");
    }
    // A mask is refused by its CONTENT, not by the flag that says a guest
    // named one. `has_user_mask` alone short-circuited this for as long as it
    // was the first half of the condition, so a guest that named a mask this
    // driver could serve exactly -- a rectangle whose true cells are the
    // diagonal -- was refused without the cells ever being read. What is still
    // refused unconditionally is a mask nobody can read: the flag is set and
    // no row is on the wire, so there is nothing to check it against.
    if plan.has_user_mask && !carries(&plan.mask_indptr) {
        return Some(
            "a user mask: attention here is causal, and a plan's mask \
                     would be dropped rather than applied",
        );
    }
    if !causal_only(plan) {
        return Some(
            "a user mask: attention here is causal, and a plan's mask \
                     would be dropped rather than applied",
        );
    }
    // A request that names SEVERAL read-out rows. This driver reads out ONE
    // per request -- `Step::readout_of` is a row per turn, and `Serving::over`
    // rewrites the sampling table to the identity because every row samples --
    // so a program that asks for a span of them runs off the end of the
    // distribution it was handed.
    //
    // What that looks like today, measured on this backend running
    // `cacheback-speculative-decoding` and `constrained-speculative-decoding`:
    //
    // ```text
    // target_tokens take: channel is poisoned: driver published poison epoch 1
    // ```
    //
    // naming no capability, no field and no driver. The reason is only in the
    // engine's log, at WARN, as `logits intrinsic row range exceeds the
    // forward's readout rows`.
    //
    // THIS GUARD DOES NOT CATCH THAT, and the comment that said it did was
    // wrong. The plans those two inferlets produce arrive with
    // `sampling_indptr=[]` -- probed under an `eprintln!` right here -- so
    // there is nothing at admission to read a width from. The width is a
    // property of the PROGRAM, whose `logits` intrinsic names the row range,
    // and the fault is raised by the reference interpreter after the fire.
    // Carrying a reason out of it and into the guest's error is an ABI
    // question -- the poison is a WORD, an epoch, with nowhere to put a
    // sentence -- and not this crate's to answer.
    //
    // The guard is kept because it is correct for the plan that states its
    // table, and it costs a comparison. It is not kept as an explanation of
    // the two failures above.
    if widest_readout(plan).is_some_and(|most| most > 1) {
        return Some(
            "a request naming several read-out rows: this driver reads out \
                     one row per request, the last one",
        );
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

/// Is every mask this plan carries the causal one this driver already
/// applies?
///
/// # Why a plan carries a mask it did not ask for
///
/// A batch of one states no masks at all. A batch that MIXES a prefill with a
/// device-resolved decode cannot, because the bridge's mask view is one
/// flattened row per query row, so the engine SYNTHESISES a row for every
/// request that had none: `RunMask::all_true(pos + 1)` -- attend everything up
/// to and including yourself, which is the definition of causal and exactly
/// what `turns.rs` does.
///
/// Refusing them because they were present refused every batch above two
/// conversations, in the words of a feature nobody had asked for. Serving them
/// because they were present would be worse: a real restriction would be
/// dropped silently. So the runs are READ, and a mask is served when it says
/// what this driver would have done anyway.
///
/// # Per request, not per row
///
/// `masks` is NOT one row per query row of the batch. The decode members are
/// elided -- a device-resolved step states its own geometry -- so a frame that
/// mixes one prefill with seven decodes carries the prefill's rows and nothing
/// else, and `mask_indptr` is what says whose they are. Comparing the two
/// lengths refused exactly the mixed frames and nothing else, which is why it
/// looked like a flake: the same eight conversations pass or fail on whether
/// the scheduler happened to put a prefill beside a decode.
///
/// # What is checked
///
/// A request states either NO mask rows, or one per query row. Each row it
/// states must be causal in both halves, because either alone is satisfied by
/// a mask that is not:
///
/// * no false run inside the row -- BRLE alternates starting FALSE, so a
///   non-empty even-indexed run is only legal once the true cells are behind
///   it;
/// * the true cells are exactly the row's own position plus one of them -- an
///   all-true span SHORTER than that is a sliding window, which is a
///   restriction this driver would silently ignore.
///
/// # A row may be WIDER than its diagonal
///
/// This is the correction, and it cost three inferlets. The check used to
/// demand `total_size == position + 1`, reading a row's WIDTH as its content.
/// A guest that builds its mask as a rectangle -- `[queries, pool_len]`, which
/// is the obvious way to build one -- states row 0 as `runs=[0, 1, 47]` over a
/// 48-wide pool: one true cell, and forty-seven false ones after it. That is
/// causal. It was refused as a user mask for as long as the width was read
/// instead of the cells, and the refusal was measured on
/// `contrastive-decoding` with its window opened past the pool, where the
/// guest's own mask reduces to `key <= query` exactly.
///
/// What matters is the TRUE set: it must be `0..=position` and nothing else.
/// False cells past the diagonal are what a causal kernel does anyway, so
/// dropping them drops nothing.
fn causal_only(plan: &LaunchPlan) -> bool {
    if plan.masks.is_empty() {
        return true;
    }
    let requests = plan.qo_indptr.len().saturating_sub(1);
    // The documented default: an absent table names no masks for anybody, and
    // then the rows above belong to nobody, which is a shape this cannot speak
    // for.
    if plan.mask_indptr.len() != requests + 1 {
        return false;
    }
    for request in 0..requests {
        let rows = usize::try_from(plan.mask_indptr[request]).unwrap_or(usize::MAX)
            ..usize::try_from(plan.mask_indptr[request + 1]).unwrap_or(usize::MAX);
        if rows.is_empty() {
            continue;
        }
        let queries = usize::try_from(plan.qo_indptr[request]).unwrap_or(usize::MAX)
            ..usize::try_from(plan.qo_indptr[request + 1]).unwrap_or(usize::MAX);
        if rows.len() != queries.len() || rows.end > plan.masks.len() {
            return false;
        }
        for (row, query) in rows.zip(queries) {
            let Some(&position) = plan.position_ids.get(query) else {
                return false;
            };
            if !causal_row(&plan.masks[row], position) {
                return false;
            }
        }
    }
    true
}

/// One mask row, against the position of the query it belongs to.
///
/// The row's TRUE cells must be exactly `0..=position`. The row may be wider
/// than that -- everything past the diagonal must simply be false.
fn causal_row(mask: &driver_api::plan::EncodedMask, position: u32) -> bool {
    let want = u64::from(position) + 1;
    let mut at = 0u64;
    let mut trues = 0u64;
    for (index, &run) in mask.runs.iter().enumerate() {
        // Clamped exactly as `EncodedMask::expand_into` clamps: a run that
        // overruns the row's width names no cell past it, so a row two wide
        // whose run says three is two true cells and not three.
        let end = at.saturating_add(u64::from(run)).min(mask.total_size);
        if index % 2 == 1 {
            // A TRUE run. It may not reach past the diagonal.
            if end > want {
                return false;
            }
            trues = trues.saturating_add(end - at);
        }
        if end == mask.total_size {
            break;
        }
        at = end;
    }
    // And the diagonal must be REACHED. These two rules are the whole of it,
    // and each one alone is satisfied by a row that is not causal: the first
    // by a window, which reaches past nothing, and the second by a row shifted
    // one cell right, which has the right number of true cells in the wrong
    // place.
    //
    // Three other rules stood here and none survived. Coverage of
    // `total_size` refused a row that names its true cells and leaves its
    // false tail unspoken, which `expand_into` defines as false anyway. A ban
    // on false runs before the diagonal, and a width of at least the
    // diagonal, are both implied by these two once the runs are clamped -- and
    // a rule no mutation of it can fail a test is a claim this file cannot
    // check, which is the thing this crate has been correcting all along.
    trues == want
}

/// Does this CSR name any rows at all?
///
/// Its LAST boundary is the total, so a table of any length whose total is
/// zero holds nothing. An empty table holds nothing either.
fn carries(indptr: &[u32]) -> bool {
    indptr.last().is_some_and(|&total| total > 0)
}

/// The most read-out rows any one request in this plan names.
///
/// `None` when the table names no requests at all, which is the shape a plan
/// that states no sampling carries -- and is not the same as every request
/// naming one row. Refusing on an ABSENT table would refuse every frame whose
/// read-out this driver derives itself.
fn widest_readout(plan: &LaunchPlan) -> Option<u32> {
    if plan.sampling_indptr.len() < 2 {
        return None;
    }
    plan.sampling_indptr
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]))
        .max()
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
/// would be a lie in the record rather than a value anyone reads. The rows a
/// caller wanted are recoverable from `Step::readout_of` plus the plan.
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
        // The frame the engine writes when a prefill and a decode land
        // together: three mask rows for the prefill, NONE for the decode --
        // which states its own geometry -- and a `mask_indptr` that says so.
        let all_true = |n: u32| driver_api::plan::EncodedMask::new(vec![0, n], u64::from(n));
        let mixed = LaunchPlan {
            masks: vec![all_true(1), all_true(2), all_true(3)],
            mask_indptr: vec![0, 3, 3],
            qo_indptr: vec![0, 3, 4],
            position_ids: vec![0, 1, 2, 9],
            ..base.clone()
        };
        assert!(
            unserved_in(&mixed).is_none(),
            "the causal mask this driver already applies is refused as a user mask, \
             which is what refused a frame that mixed a prefill with a decode"
        );

        // Each half of "causal" is load-bearing, and so is the CSR. A mask
        // that is all-true over a SHORTER span is a sliding window; one with a
        // hole hides a token; a request whose rows do not cover its queries is
        // a shape this cannot speak for.
        let mut window = mixed.clone();
        window.masks[2] = all_true(2);
        assert!(
            unserved_in(&window).is_some(),
            "a sliding window is served as though it were causal"
        );
        let mut holed = mixed.clone();
        holed.masks[2] = driver_api::plan::EncodedMask::new(vec![1, 2], 3);
        assert!(
            unserved_in(&holed).is_some(),
            "a mask that hides a token is served as though it were causal"
        );
        let mut short = mixed.clone();
        short.mask_indptr = vec![0, 2, 3];
        assert!(
            unserved_in(&short).is_some(),
            "a request whose mask rows do not cover its query rows is read anyway"
        );

        // A rectangle, which is how a guest that builds a mask by hand builds
        // it: `[queries, pool_len]`, so every row is the pool's width and the
        // cells past the diagonal are false. These three rows are the ones
        // `contrastive-decoding` puts on the wire with its window opened past
        // the pool -- row 0 arrived as `runs=[0, 1, 47]`, `total_size=48`,
        // measured under an `eprintln!` in this function. Their true cells are
        // exactly the diagonal, so they are causal, and refusing them refused
        // an inferlet this driver can serve.
        let padded = |n: u32, width: u64| {
            driver_api::plan::EncodedMask::new(vec![0, n, u32::try_from(width).unwrap() - n], width)
        };
        let rectangle = LaunchPlan {
            masks: vec![padded(1, 48), padded(2, 48), padded(3, 48)],
            mask_indptr: vec![0, 3, 3],
            qo_indptr: vec![0, 3, 4],
            position_ids: vec![0, 1, 2, 9],
            has_user_mask: true,
            ..base.clone()
        };
        assert!(
            unserved_in(&rectangle).is_none(),
            "a causal mask padded to the pool's width is refused as a user \
             mask, which is what refused `contrastive-decoding` at every window"
        );

        // Each half of the correction is load-bearing. A rectangle whose true
        // span is SHORT of the diagonal is a sliding window wearing the same
        // shape -- and it is the shape the very same inferlet puts on the wire
        // at its default window of 8, so this is not a hypothetical row.
        let mut narrow = rectangle.clone();
        narrow.masks[2] = padded(2, 48);
        assert!(
            unserved_in(&narrow).is_some(),
            "a sliding window padded to the pool's width is served as though \
             it were causal, which is the failure the padding rule risks"
        );
        // And a true cell PAST the diagonal is a row attending to its own
        // future, which the width rule used to make unrepresentable. The
        // count alone does not catch this one: the row is SHIFTED, so it has
        // exactly as many true cells as a causal row and every one of them is
        // in the wrong place.
        let mut ahead = rectangle.clone();
        ahead.masks[2] = driver_api::plan::EncodedMask::new(vec![1, 3, 44], 48);
        assert!(
            unserved_in(&ahead).is_some(),
            "a row shifted one cell right attends to its own future and is \
             served as causal, because it counts the same"
        );
        // A row narrower than its own diagonal is a window stated as a width,
        // and the clamp is what makes the count see it: three true cells over
        // a row two wide are two true cells.
        let mut narrow_row = rectangle.clone();
        narrow_row.masks[2] = driver_api::plan::EncodedMask::new(vec![0, 3], 2);
        assert!(
            unserved_in(&narrow_row).is_some(),
            "a row too narrow to hold its own history is served as causal"
        );
        // A hole is still a hole at any width.
        let mut holed_wide = rectangle.clone();
        holed_wide.masks[2] = driver_api::plan::EncodedMask::new(vec![1, 2, 45], 48);
        assert!(
            unserved_in(&holed_wide).is_some(),
            "a mask that hides a token is served once the row is padded"
        );
        // A row that STOPS short is the case the count rule exists for, and
        // the reason coverage of `total_size` could not stand in for it: the
        // runs `[0, 1]` over a 48-wide row are one true cell and forty-seven
        // false ones, which at position 2 hides two tokens the query may see.
        // Nothing about its width says so.
        let mut stops_short = rectangle.clone();
        stops_short.masks[2] = driver_api::plan::EncodedMask::new(vec![0, 1], 48);
        assert!(
            unserved_in(&stops_short).is_some(),
            "a row whose true cells stop short of its own position is served \
             as causal, so a window encoded economically is dropped"
        );
        // And the economy itself is legal: a row that names its true cells and
        // nothing after them is the same row as one that spells the false tail
        // out, because `expand_into` writes only what the runs name.
        let mut economical = rectangle.clone();
        economical.masks[2] = driver_api::plan::EncodedMask::new(vec![0, 3], 48);
        assert!(
            unserved_in(&economical).is_none(),
            "a causal row that leaves its false tail unspoken is refused for \
             not spelling out cells the encoding defines as false"
        );
        // The flag alone still refuses, because a mask with no rows behind it
        // is a mask nobody can read. This is the case that must NOT be lost
        // when the flag stops short-circuiting.
        assert!(
            unserved_in(&LaunchPlan {
                has_user_mask: true,
                ..base.clone()
            })
            .is_some(),
            "a plan that says it carries a user mask and shows no rows is \
             served on the strength of showing nothing"
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

        // A read-out of several rows for one request. A plan that STATES its
        // sampling table is refused for it -- though the two speculative
        // decoders that need it do not state one, so this guard is not what
        // answers them; see the note beside it.
        assert_eq!(
            unserved_in(&LaunchPlan {
                sampling_indptr: vec![0, 3],
                ..base.clone()
            }),
            Some(
                "a request naming several read-out rows: this driver reads out \
                 one row per request, the last one"
            ),
            "a request asking for three distributions is launched, and faults \
             inside the interpreter instead of being refused by name"
        ); // Both halves of the boundary. One row per request is the ordinary
        // shape and must be served; a table that names nothing at all is a
        // plan whose read-out this driver derives, and must be served too --
        // refusing on an absent table would refuse every frame there is.
        assert!(
            unserved_in(&LaunchPlan {
                sampling_indptr: vec![0, 1, 2],
                ..base.clone()
            })
            .is_none(),
            "one read-out row per request is refused, so this driver serves \
             no ordinary frame"
        );
        assert!(
            unserved_in(&LaunchPlan {
                sampling_indptr: vec![],
                ..base.clone()
            })
            .is_none(),
            "a plan that states no sampling table is refused for stating none"
        );
    }
}
