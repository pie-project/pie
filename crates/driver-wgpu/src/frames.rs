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
///
/// # Why the steps' own page lists are read too
///
/// Because those are the pages this driver will BIND, and the other two
/// sources are the engine's statements ABOUT them. Both statements can be
/// short of the fact: `required_kv_pages` is a declared high-water, the
/// frame's translation is a placement table that is empty whenever nothing
/// was moved, and only one of the engine's two batch-assembly paths folds
/// `kv_page_indices` into either. A frame whose page list ran past both was
/// answered with `NoSuchPage` by `Request::stage` -- a correct bounds check,
/// on a pool that had simply never been told to grow.
///
/// Strictly more permissive and never less: the declarations are still maxed
/// in, so a frame that declares more than it binds is unaffected.
///
/// A DEVICE-GEOMETRY step binds pages this cannot see, because its list
/// arrives on channels and is translated by `envelope::fill` long after
/// admission. Those are covered by the translation, which is that class's
/// placement table and is never empty for it -- the two sources are
/// complementary rather than redundant.
#[must_use]
pub fn pages_named(frame: &FrameSubmission) -> u32 {
    frame
        .kv_translation
        .iter()
        .chain(
            frame
                .steps
                .iter()
                .flat_map(|step| step.plan.kv_page_indices.iter()),
        )
        .copied()
        .filter(|&p| p != u32::MAX)
        .max()
        .map_or(0, |p| p.saturating_add(1))
        .max(frame.required_kv_pages)
}

/// The pages a KV copy plan writes INTO, as a count the pool must cover.
///
/// [`pages_named`]'s sibling, and here beside it because they answer the same
/// question about the same elastic pool through two different doors: a frame
/// arrives at [`crate::shell::Shell::admit`] and a copy plan at
/// [`crate::shell::Shell::copy_kv`], and a pool that grows for one and not
/// the other refuses work it could serve.
///
/// # Destinations only
///
/// A source above the pool stays a refusal, and must. This pool only ever
/// grows on demand, so a page number it has never held is a page nothing has
/// ever written; growing for it would turn a refusal into a copy of freshly
/// zeroed memory, which is history-shaped silence rather than an error.
#[must_use]
pub fn copy_pages_named(plan: &driver_api::KvCopyPlan) -> u32 {
    plan.dst_page_ids
        .iter()
        .copied()
        .chain(plan.cells.iter().map(|cell| cell.dst_page_id))
        .max()
        .map_or(0, |page| page.saturating_add(1))
}

/// Every row of every request writes inside the span the engine declared for
/// it.
///
/// `kv_write_lower_bounds`/`upper_bounds` are one pair per request, and they
/// arrive on 490 plans of a single curated sweep -- the engine sets them for
/// every pipeline-scoped fire, `writable_pages.start * page_size` and `.end *
/// page_size`. **No driver reads them.** Not this one until now, and not
/// `driver-metal`, `driver-vulkan` or `driver-cuda`, so the declaration had
/// never been checked by anybody.
///
/// What it guards is this crate's headline defect: a fire that writes into a
/// page the scheduler reserved for somebody else does not fault, it answers
/// fluently with another conversation's history mixed in. The engine checks
/// the READ side itself -- "KV read page {page} escapes the readable
/// declaration" -- and hands the write side to a driver.
///
/// # The coordinate, which I got wrong first
///
/// LOGICAL slots, in the working set's own numbering, not physical ones.
/// Comparing a physical slot against this span reports 4652 violations on a
/// clean sweep, all of them the comparison's: a request whose first page is
/// physical 7 writes physical slot 112 for its position 0, and 112 is outside
/// a `0..64` span that means logical pages 0..4.
///
/// So a DERIVED write is at its position, which is the logical slot by
/// definition, and a STATED one is at `w_page * page_size + w_off` in the same
/// numbering -- `w_page` is a working-set index, translated by
/// `envelope::fill` like every other page of that class.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] naming the request, the row, the slot and the
/// span. A request the engine stated no span for is unbounded, which is what
/// the scheduler pushes for one that declared nothing: `(0, u64::MAX)`.
pub fn writes_stay_in_the_declared_span(
    plan: &LaunchPlan,
    requests: &[crate::resources::Request],
    page_size: u32,
) -> Result<(), Unlaunched> {
    if plan.kv_write_lower_bounds.is_empty() {
        return Ok(());
    }
    let page_size = u64::from(page_size);
    for (r, request) in requests.iter().enumerate() {
        let lower = plan.kv_write_lower_bounds.get(r).copied().unwrap_or(0);
        let upper = plan
            .kv_write_upper_bounds
            .get(r)
            .copied()
            .unwrap_or(u64::MAX);
        for (j, &position) in request.positions.iter().enumerate() {
            let slot = match request.writes.get(j).and_then(|w| *w) {
                Some((page, offset)) => u64::from(page) * page_size + u64::from(offset),
                None => u64::from(position),
            };
            if slot < lower || slot >= upper {
                return Err(Unlaunched::Malformed(format!(
                    "request {r} row {j} writes slot {slot} and the engine declared \
                     it may write {lower}..{upper}"
                )));
            }
        }
    }
    Ok(())
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
/// # Why ONE field per family is enough, which was audited rather than assumed
///
/// `LaunchPlan` has 57 fields and this function names eleven. Twenty-seven are
/// not mentioned anywhere in this crate, and most of them belong to a family
/// whose REPRESENTATIVE is refused here: `image_pixels` stands for eight image
/// fields, `rs_slot_ids` for eleven recurrent ones, `embed_rows` for five.
///
/// That is only sound if a secondary can never arrive without its
/// representative, which is a claim about the engine and not about this file.
/// It was measured: a probe over a full curated sweep, asking for every family
/// whether any secondary carried something while the representative was empty.
/// **Zero.**
///
/// The structural reason, which is why it should stay true: the representative
/// is the DATA and the secondaries DESCRIBE it. `image_grids` is the shape of
/// the pixels, `embed_dtypes` the type of the rows, `rs_fold_lens` the extent
/// of the slots. There is nothing to describe when the thing is not there.
///
/// The probe found 235 apparent violations first, and all of them were the
/// probe's: an all-zero CSR is present-but-empty, and `indptr.len() > 1` is
/// the wrong test for it. `carries` is the right one and this function
/// already uses it -- which is the whole reason it exists.
///
/// Two fields were genuinely uncovered and are handled: `kv_len_device` is
/// refused below, and `kv_write_lower_bounds`/`upper_bounds` are ENFORCED by
/// [`writes_stay_in_the_declared_span`] rather than refused, because this
/// driver can honour them. `context_ids` is uncovered and inert -- nothing in
/// `engine` or `worker` ever populates it, and it was empty on every plan of
/// the sweep.
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
pub fn unserved_in(plan: &LaunchPlan, holds_recurrent: bool) -> Option<&'static str> {
    // Order is the order the fields appear in `LaunchPlan`, so that a field
    // added there and not here is visible as a gap in this list rather than
    // as an answer nobody checked.
    // A kv_len that lives ON THE DEVICE, which this driver cannot read.
    //
    // The scalar `kv_len` beside it is what this driver uses, and the wire's
    // own comment says which is authoritative: `kv_len_device[0]` is the base
    // of a packed `[R]` u32 device buffer holding request `r`'s length, and
    // "empty implies all lanes host-fed via the scalar `kv_len` above". So a
    // non-empty handle means the scalar is NOT the answer, and a driver that
    // reads it anyway attends the wrong amount of history -- fluently, because
    // a shorter or longer span of real keys is still real keys.
    //
    // This driver's own device-geometry class gets its lengths from CHANNELS,
    // through `envelope::fill`, which is the portable way to say the same
    // thing. This is the other way and it is not implemented.
    //
    // Measured before adding: EMPTY on every plan of a full curated sweep, so
    // this refuses a path nothing currently takes -- which is the point.
    // `max_layers` and `hook_page_mask` are refused on the same grounds.
    if !plan.kv_len_device.is_empty() {
        return Some(
            "`kv_len_device`: a device-resident KV length this driver cannot read, so \
             its scalar `kv_len` would be the wrong answer rather than a missing one",
        );
    }
    // `kv_translation_version` is deliberately NOT here. It arrives non-zero on
    // 469 plans of a sweep and it is a STAMP, not an instruction: the engine
    // and the worker use it to tell one translation from another, and a driver
    // acts on the translation it was handed rather than on which one it is.
    // Refusing it would refuse every pipeline-scoped fire.
    // A question about the DRIVER and not about the plan, which is why it is
    // the one refusal here that takes an argument. This read *"no model this
    // driver serves holds any"* and was true when written; a deployment that
    // states a `resources::Recurrent` shape now opens a `RecurrentPool`, and
    // refusing its plan would refuse the thing the pool was allocated for.
    //
    // Still refused when there is no pool, and emphatically: the slot ids are
    // where each request's carry LIVES, so serving the plan without them
    // would run every request against slot zero -- one carry shared by all of
    // them, which is not a fault and not a NaN.
    if !holds_recurrent && (!plan.rs_slot_ids.is_empty() || !plan.rs_buffer_slot_ids.is_empty()) {
        return Some("recurrent state: this deployment allocated no slots for it");
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
    // A mask is APPLIED now, so the only thing left to refuse is one nobody
    // can read: the flag is set and no row is on the wire, which leaves
    // nothing to build a rectangle from.
    //
    // What used to be here refused any mask that was not exactly causal. That
    // was honest while the tables were staged as zeros -- a mask that would be
    // dropped is worse than a refusal -- and it is now the wrong answer:
    // `requests_of` decodes each row's runs into allow-bytes, `Frame::of`
    // packs them into a rectangle as wide as the fire's widest row, and
    // `attn/sdpa_paged.wgsl` reads `attention_mask[row * stride + kp]` for
    // every row whose enable byte is set.
    //
    // The causal reader stays, because it is still the thing that says a
    // SYNTHESIZED mask and no mask are the same computation -- see
    // `masks_are_exactly_causal`'s own note. It is no longer a gate.
    if plan.has_user_mask && !carries(&plan.mask_indptr) {
        return Some(
            "a user mask with no rows on the wire: the flag is set and \
                     there is nothing to read it from",
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
    // A request naming SEVERAL readout rows is served now. It used to be
    // refused here, because `Step::readout_of` was one row per turn and the
    // rows a caller asked for could not be found again -- `Serving::over`
    // forces every row to sample, so they always EXISTED, and what was
    // missing was the mapping from a request to its row SPAN. `Frame` records
    // `sampling_indptr` now and `Step::readouts_of` is that mapping.
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
    // `dense_device_mask` is RESOLVED now, not refused. `envelope::fill` reads
    // the dense bytes off the program's resolution, re-encodes them as runs
    // and clears the flag, so a plan still carrying it here is one that never
    // went through that path -- a wire plan claiming the driver resolved
    // something it was never handed.
    if plan.dense_device_mask {
        return Some(
            "`dense_device_mask` on a plan this driver did not resolve: the \
                     flag says a dense per-cell mask was read from a channel, \
                     and no channel was",
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
#[cfg(test)]
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
#[cfg(test)]
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
        let (lo, hi) = member_requests(&sub.program_row_indptr, member, step.readout_of.len())
            .ok_or_else(|| {
                Unlaunched::Malformed(format!(
                    "instance {id} is member {member}, which the step's {}-entry attribution \
                     CSR does not describe over {} request(s)",
                    sub.program_row_indptr.len(),
                    step.readout_of.len()
                ))
            })?;
        let mut values = Vec::with_capacity((hi - lo) * vocab);
        let mut rows = 0usize;
        for r in lo..hi {
            // Every row this request named, not just its last. A decode names
            // one and this is the loop it always was; a speculative verifier
            // names one per drafted token, and handing it the last row alone
            // is a different answer rather than a smaller one -- the program
            // reads a row range and faults inside the interpreter.
            //
            // `readouts_of` is the frame's own grouping, so a row here is a
            // row of THIS request. A neighbour's row would be a real
            // distribution belonging to another conversation, which
            // `frames::sample_rows_of` refuses at conversion.
            // The empty case falls back to the request's LAST row, which is
            // what a decode means and what `readout_of` is. A turn that
            // contributed NO rows has no last row, and `turns::last_row_of`
            // says so with `NO_ROW` rather than answering `0` -- the first
            // turn's -- so the bound check below refuses it by name instead of
            // handing over another conversation's distribution.
            let span = step.readouts_of.get(r).map_or(&[][..], Vec::as_slice);
            let span: &[usize] = if span.is_empty() {
                std::slice::from_ref(&step.readout_of[r])
            } else {
                span
            };
            for &at in span {
                let one = step.logits.row(at).ok_or_else(|| {
                    Unlaunched::Malformed(format!(
                        "request {r} reads row {at} of a read-out of {} rows",
                        step.logits.rows
                    ))
                })?;
                values.extend_from_slice(one);
                rows += 1;
            }
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
                rows: rows as u32,
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
/// # Why an unusable CSR is `None` rather than the same fallback
///
/// It used to be the same fallback, and that reproduces the defect above for
/// exactly the member the frame got wrong. On `[0, 1, 2, 9]` with three
/// requests, members 0 and 1 answer `(0, 1)` and `(1, 2)` -- correct -- and
/// member 2 answers `(0, 3)`, taking all three requests' rows and sampling
/// request 0's distribution first. One member out of three, silently
/// answering another conversation, in a frame whose other members are fine. A
/// CSR too short for the roster does the same.
///
/// "Absent" and "present but not describing this member" are different
/// claims. The first is a frame that states no attribution and means the
/// single-member case; the second is a frame whose tables disagree with its
/// own roster, which is what both callers already refuse by name a few lines
/// earlier. So this returns `None` and they refuse it too.
///
/// `driver-vulkan` found this and its `member_requests` says the same thing;
/// this copy answered for another eleven weeks.
#[must_use]
pub fn member_requests(
    program_row_indptr: &[u32],
    member: usize,
    requests: usize,
) -> Option<(usize, usize)> {
    if program_row_indptr.len() < 2 {
        return Some((0, requests));
    }
    match (
        program_row_indptr.get(member),
        program_row_indptr.get(member + 1),
    ) {
        (Some(&s), Some(&e)) if e >= s && e as usize <= requests => Some((s as usize, e as usize)),
        _ => None,
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
        let mut request = Request::of(
            plan.position_ids[lo..hi].to_vec(),
            plan.kv_page_indices[plo..phi].to_vec(),
        );
        request.mask = mask_rows_of(plan, r, hi - lo)?;
        request.samples = sample_rows_of(plan, r, lo, hi)?;
        // THE ENGINE'S SEAT, not this driver's. `rs_slot_ids` is one per
        // request and it is the engine that assigns them across fires; a
        // driver that renumbered them would point a scan at another
        // conversation's carry. `Frame::of` spreads it per row, which is the
        // same table `driver-metal` builds as `rs_slot_ids[req_of_token[t]]`.
        //
        // Empty for a model with no recurrent state, where nothing reads it.
        // A plan that states SOME slots and not enough of them is malformed
        // rather than defaulted: slot zero is a real slot holding a real
        // conversation, so the default would be another request's carry, read
        // fluently and silently.
        if !plan.rs_slot_ids.is_empty() {
            let Some(&slot) = plan.rs_slot_ids.get(r) else {
                return Err(Unlaunched::Malformed(format!(
                    "the plan describes {rows} requests and states {} recurrent slots",
                    plan.rs_slot_ids.len()
                )));
            };
            request.slot = slot;
        }
        out.push(request);
    }
    Ok(out)
}

/// The rows request `r` reads out, in ITS OWN row numbering.
///
/// The plan states them globally -- `validate_geometry` checks each against the
/// fire's total rows -- and `resources::Request` states them per request, which
/// is what lets `Frame::of` place them after seriation. The subtraction is the
/// whole conversion.
///
/// Empty for a request naming none, which `Request::read` reads as "the last
/// row" -- the decode case, and the overwhelming majority.
///
/// # Why this exists at all
///
/// It used to be dropped, and a request naming SEVERAL readout rows was
/// refused by name. The rows always existed -- `Serving::over` forces every
/// row to sample -- and what was missing was exactly this: a mapping from a
/// request to its row SPAN. A speculative verifier names one row per drafted
/// token, so without it `cacheback-speculative-decoding` could not run.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] for a CSR that does not close or a row outside
/// the request that claims it. A row from a NEIGHBOUR's span would read a real
/// distribution belonging to another conversation, which is the failure this
/// conversion has to make impossible rather than likely.
fn sample_rows_of(
    plan: &LaunchPlan,
    r: usize,
    lo: usize,
    hi: usize,
) -> Result<Vec<u32>, Unlaunched> {
    if plan.sampling_indptr.is_empty() {
        return Ok(Vec::new());
    }
    if plan.sampling_indptr.len() != plan.qo_indptr.len() {
        return Err(Unlaunched::Malformed(format!(
            "sampling_indptr has {} boundaries and qo_indptr has {}",
            plan.sampling_indptr.len(),
            plan.qo_indptr.len()
        )));
    }
    let (slo, shi) = (
        plan.sampling_indptr[r] as usize,
        plan.sampling_indptr[r + 1] as usize,
    );
    if slo > shi || shi > plan.sampling_indices.len() {
        return Err(Unlaunched::Malformed(format!(
            "request {r} spans readouts {slo}..{shi} of {}",
            plan.sampling_indices.len()
        )));
    }
    let mut out = Vec::with_capacity(shi - slo);
    // Numbered within the REQUEST, the whole way from the scheduler to
    // `Frame::of` -- which is the one place the numbering changes, because
    // that is where the fire's rows are laid out. `envelope::fill` keeps it
    // too, and `driver-vulkan` reads and writes the same thing at both
    // stages; a driver that rebased here and there would be right only while
    // every member had one request.
    let width = hi - lo;
    for &row in &plan.sampling_indices[slo..shi] {
        let row = row as usize;
        if row >= width {
            return Err(Unlaunched::Malformed(format!(
                "request {r} reads out its own row {row}, past the {width} row(s) it \
                 spans ({lo}..{hi})"
            )));
        }
        out.push(u32::try_from(row).unwrap_or(u32::MAX));
    }
    Ok(out)
}

/// One request's mask, decoded from its BRLE runs into allow-bytes.
///
/// Empty when the request states none, which is not a mask of zeros: the first
/// leaves the row's enable byte clear and lets the causal rule alone apply,
/// the second forbids every key.
///
/// # Why the decode is here and not in `resources`
///
/// `resources` is the portable half and `EncodedMask` is a wire type. This
/// module already reads the same runs for [`masks_are_exactly_causal`], so the
/// two readings live side by side where they can be compared.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] for a CSR that does not close, a run list that
/// does not cover its own `total_size`, or a request whose mask names a
/// different number of rows than it contributes. Every one of those would
/// otherwise index one row's mask at another row's offset.
fn mask_rows_of(plan: &LaunchPlan, r: usize, rows: usize) -> Result<Vec<Vec<u8>>, Unlaunched> {
    if plan.mask_indptr.len() != plan.qo_indptr.len() {
        // No table at all is the ordinary case; a table of the wrong length is
        // a plan nobody can read, and reading it anyway would take another
        // request's runs.
        if plan.mask_indptr.is_empty() {
            return Ok(Vec::new());
        }
        return Err(Unlaunched::Malformed(format!(
            "mask_indptr has {} boundaries and qo_indptr has {}",
            plan.mask_indptr.len(),
            plan.qo_indptr.len()
        )));
    }
    let (lo, hi) = (
        plan.mask_indptr[r] as usize,
        plan.mask_indptr[r + 1] as usize,
    );
    if lo == hi {
        return Ok(Vec::new());
    }
    if lo > hi || hi > plan.masks.len() {
        return Err(Unlaunched::Malformed(format!(
            "request {r} spans masks {lo}..{hi} of {}",
            plan.masks.len()
        )));
    }
    if hi - lo != rows {
        return Err(Unlaunched::Malformed(format!(
            "request {r} states {} masks and contributes {rows} rows",
            hi - lo
        )));
    }
    plan.masks[lo..hi].iter().map(decode_mask).collect()
}

/// [`decode_mask`], for `envelope`'s round-trip check.
///
/// Public to the crate rather than duplicated there: the point of that test is
/// that the encoder and THIS reader agree, and a second reader written beside
/// the encoder would only check it against itself.
#[cfg(test)]
pub(crate) fn decode_mask_for_test(
    mask: &driver_api::plan::EncodedMask,
) -> Result<Vec<u8>, Unlaunched> {
    decode_mask(mask)
}

/// One BRLE mask into one byte per key.
///
/// The encoding alternates run lengths starting with a FALSE run, which is
/// what [`all_true`] reads when it checks that every even-indexed run is
/// empty. Same reading, kept next to it.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] when the runs do not cover `total_size`. A short
/// run list would leave the tail of the row at whatever the vector was
/// initialised with, which is "forbidden" here and would be a plausible mask
/// nobody wrote.
fn decode_mask(mask: &driver_api::plan::EncodedMask) -> Result<Vec<u8>, Unlaunched> {
    let total = usize::try_from(mask.total_size)
        .map_err(|_| Unlaunched::Malformed("a mask longer than this host can address".into()))?;
    let mut out = Vec::with_capacity(total);
    for (index, &run) in mask.runs.iter().enumerate() {
        let byte = u8::from(index % 2 == 1);
        for _ in 0..run {
            if out.len() == total {
                break;
            }
            out.push(byte);
        }
    }
    if out.len() != total {
        return Err(Unlaunched::Malformed(format!(
            "a mask's runs cover {} of its {total} keys",
            out.len()
        )));
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

    /// Each member owns its own slice of the read-out, and no other.
    #[test]
    fn a_member_owns_the_requests_its_frame_attributes_to_it() {
        assert_eq!(super::member_requests(&[0, 1, 3], 0, 3), Some((0, 1)));
        assert_eq!(
            super::member_requests(&[0, 1, 3], 1, 3),
            Some((1, 3)),
            "a speculative member owns several"
        );
    }

    /// A frame stating no attribution means the single-member case.
    #[test]
    fn an_absent_attribution_gives_the_one_member_the_whole_read_out() {
        assert_eq!(super::member_requests(&[], 0, 3), Some((0, 3)));
        assert_eq!(super::member_requests(&[0], 0, 3), Some((0, 3)));
    }

    /// A CSR that does not place a member is refused, not fallen back on.
    ///
    /// The fallback was `(0, requests)`, and for `[0, 1, 2, 9]` over three
    /// requests that gave member 2 all three -- so it sampled request 0's
    /// distribution while members 0 and 1 stayed correct. One member of a
    /// frame answering another conversation, with nothing faulting. That is
    /// the `driver-metal` defect this function's own doc describes, reached
    /// through the function meant to prevent it.
    #[test]
    fn a_member_the_attribution_does_not_place_is_refused() {
        assert_eq!(
            super::member_requests(&[0, 1, 2, 9], 2, 3),
            None,
            "the last entry runs past the read-out"
        );
        assert_eq!(
            super::member_requests(&[0, 1], 2, 3),
            None,
            "a roster longer than the CSR that describes it"
        );
        assert_eq!(
            super::member_requests(&[0, 2, 1], 1, 3),
            None,
            "and a window that ends before it starts"
        );
    }

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

    /// **The engine's recurrent seats are carried through, and a short list is
    /// refused rather than defaulted.**
    ///
    /// `rs_slot_ids` is the engine's assignment of a conversation to a place
    /// in the gated DeltaNet's state slab, and it is the ENGINE's because it
    /// holds across fires. A driver that dropped it would put every
    /// conversation in slot zero — which is not an empty slot, it is the first
    /// conversation's — and the model would answer fluently out of somebody
    /// else's carry.
    ///
    /// That is not hypothetical. `Frame::recurrent_slots` was declared,
    /// staged and never written, and the visible symptom was that the same
    /// prompt answered a different way every time it was asked.
    ///
    /// So the short-list case refuses instead of padding with zero, for the
    /// same reason: the pad would be a real seat.
    #[test]
    fn the_engine_states_which_recurrent_seat_each_request_holds() {
        let mut p = plan(&[0, 2, 3], &[1, 2, 3], &[0, 1, 0], &[0, 1], &[0, 1, 2]);
        p.rs_slot_ids = vec![5, 2];
        let got = requests_of(&p).expect("two requests");
        assert_eq!(
            got.iter().map(|r| r.slot).collect::<Vec<_>>(),
            vec![5, 2],
            "the engine's seats were not carried onto the requests"
        );

        // A model with no recurrent state states none, and nothing reads it.
        let none = plan(&[0, 2, 3], &[1, 2, 3], &[0, 1, 0], &[0, 1], &[0, 1, 2]);
        assert_eq!(
            requests_of(&none)
                .expect("two requests")
                .iter()
                .map(|r| r.slot)
                .collect::<Vec<_>>(),
            vec![0, 0],
        );

        let mut short = plan(&[0, 2, 3], &[1, 2, 3], &[0, 1, 0], &[0, 1], &[0, 1, 2]);
        short.rs_slot_ids = vec![5];
        let Err(Unlaunched::Malformed(why)) = requests_of(&short) else {
            panic!("a plan stating one seat for two requests was accepted");
        };
        assert!(
            why.contains('2') && why.contains('1'),
            "the refusal does not say how many of each: {why}"
        );
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

    /// ...and by the pages its STEPS bind, which the declarations can be
    /// short of.
    ///
    /// `kv_translation` and `required_kv_pages` are both the engine's
    /// statements ABOUT a frame's pages; `kv_page_indices` is the list this
    /// driver hands to `Request::stage`. They differ: the translation is
    /// empty whenever nothing was moved, and only one of the engine's two
    /// batch-assembly paths folds the page list into the declared high-water.
    /// A frame that ran past both was refused by `NoSuchPage` -- a request
    /// killed for a page the scheduler was entitled to hand out and the pool
    /// could have grown to hold, which is the fault `admit` exists to
    /// prevent.
    #[test]
    fn a_frame_is_also_sized_by_the_pages_its_steps_bind() {
        let step = |pages: &[u32]| driver_api::StepSubmission {
            plan: LaunchPlan {
                kv_page_indices: pages.to_vec(),
                ..LaunchPlan::default()
            },
            roster_rows: Vec::new(),
            sub_batch_indptr: vec![0],
            sub_batch_class: Vec::new(),
            terminal_cells: Vec::new(),
            program_row_indptr: Vec::new(),
            logical_fire_ids: Vec::new(),
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: Vec::new(),
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        };
        // Nothing was moved, so the translation is empty, and the declared
        // high-water is short of the page the second step binds.
        let frame = FrameSubmission {
            instance_ids: vec![1],
            kv_translation: Vec::new(),
            kv_translation_indptr: vec![0, 0],
            required_kv_pages: 2,
            steps: vec![step(&[0, 1]), step(&[7])],
        };
        assert_eq!(
            pages_named(&frame),
            8,
            "page 7 is a page this driver will BIND, so it is a page the pool \
             must be grown to hold"
        );

        // Every step, not just the first: a frame's later steps are exactly
        // where the pages the earlier ones did not need show up.
        let frame = FrameSubmission {
            steps: vec![step(&[0]), step(&[3]), step(&[11])],
            ..frame
        };
        assert_eq!(pages_named(&frame), 12);

        // And the declaration still wins when it is the larger claim, because
        // folding the lists in is strictly more permissive and never less.
        let frame = FrameSubmission {
            required_kv_pages: 64,
            ..frame
        };
        assert_eq!(pages_named(&frame), 64);
    }

    /// A row outside the engine's declared writable span is refused, and the
    /// coordinate is the LOGICAL one.
    ///
    /// The three cases that matter, in one fixture: a derived write inside the
    /// span, a derived write outside it, and a STATED write whose logical slot
    /// is inside while the physical page it names is far outside. That last
    /// one is the case a physical-coordinate comparison gets wrong, and it is
    /// how 4652 false violations were counted on a clean sweep before the
    /// coordinate was understood.
    #[test]
    fn a_row_outside_the_engines_declared_write_span_is_refused() {
        use crate::resources::Request;

        let plan = |lo: Vec<u64>, hi: Vec<u64>| LaunchPlan {
            kv_write_lower_bounds: lo,
            kv_write_upper_bounds: hi,
            ..LaunchPlan::default()
        };
        // Positions 16..19, which is logical page 1 at a page size of 16.
        let inside = Request::of(vec![16, 17, 18], vec![9, 4]);

        assert!(
            writes_stay_in_the_declared_span(
                &plan(vec![16], vec![32]),
                std::slice::from_ref(&inside),
                16
            )
            .is_ok(),
            "rows 16..19 are inside a 16..32 span"
        );

        let Err(Unlaunched::Malformed(why)) = writes_stay_in_the_declared_span(
            &plan(vec![0], vec![18]),
            std::slice::from_ref(&inside),
            16,
        ) else {
            panic!("position 18 is past an 0..18 span");
        };
        assert!(
            why.contains("row 2") && why.contains("18") && why.contains("0..18"),
            "the refusal names neither the row, the slot nor the span: {why}"
        );

        // A STATED write: logical page 1 offset 3 is slot 19, inside 16..32 --
        // and the PHYSICAL page it lands on is 4, which as a slot would be 64
        // and outside. The check must read the logical one.
        let stated = Request {
            writes: vec![Some((1, 3)), Some((1, 4)), Some((1, 5))],
            ..Request::of(vec![16, 17, 18], vec![9, 4])
        };
        assert!(
            writes_stay_in_the_declared_span(
                &plan(vec![16], vec![32]),
                std::slice::from_ref(&stated),
                16
            )
            .is_ok(),
            "a stated write at logical page 1 is inside a 16..32 span, whatever \
             physical page it is placed on"
        );

        // ...and the STATED slot is what is checked, not the position beside
        // it. Positions 16..19 are inside a 16..32 span; a descriptor putting
        // them on logical page 0 writes slots 0..3, which is not. A check
        // reading the position would pass this, which is what the first
        // version of this test failed to distinguish.
        let elsewhere = Request {
            writes: vec![Some((0, 0)), Some((0, 1)), Some((0, 2))],
            ..Request::of(vec![16, 17, 18], vec![9, 4])
        };
        let Err(Unlaunched::Malformed(why)) = writes_stay_in_the_declared_span(
            &plan(vec![16], vec![32]),
            std::slice::from_ref(&elsewhere),
            16,
        ) else {
            panic!("a descriptor aimed at logical page 0 is outside a 16..32 span");
        };
        assert!(
            why.contains("slot 0") && why.contains("16..32"),
            "the refusal reads the position rather than the stated slot: {why}"
        );

        // A request the engine stated no span for is unbounded.
        assert!(
            writes_stay_in_the_declared_span(
                &plan(vec![16], vec![32]),
                &[inside.clone(), Request::of(vec![900], vec![1])],
                16
            )
            .is_ok(),
            "the second request has no pair of its own and is not bounded by \
             the first request's"
        );
        // ...and a plan that declares nothing bounds nothing.
        assert!(
            writes_stay_in_the_declared_span(
                &LaunchPlan::default(),
                std::slice::from_ref(&inside),
                16
            )
            .is_ok()
        );
    }

    /// A copy plan is counted by where it WRITES, and never by where it
    /// reads.
    ///
    /// The cells count as well as the whole-page moves: a prefix share can be
    /// a single row into a page the pool has not grown to, and a `need` taken
    /// over `dst_page_ids` alone would miss it.
    #[test]
    fn a_copy_plan_is_sized_by_its_destinations_and_not_its_sources() {
        let plan = driver_api::KvCopyPlan {
            src_page_ids: vec![90, 91],
            dst_page_ids: vec![2, 5],
            ..driver_api::KvCopyPlan::default()
        };
        assert_eq!(
            copy_pages_named(&plan),
            6,
            "the pool must reach page 5 to be written to, and never reaches \
             page 91 to be read from"
        );

        let with_cell = driver_api::KvCopyPlan {
            cells: vec![driver_api::local::KvMoveCell {
                src_page_id: 80,
                src_token_offset: 0,
                dst_page_id: 11,
                dst_token_offset: 3,
            }],
            ..plan
        };
        assert_eq!(
            copy_pages_named(&with_cell),
            12,
            "a single-row share into page 11 needs page 11 as much as a whole \
             page move does"
        );

        assert_eq!(
            copy_pages_named(&driver_api::KvCopyPlan::default()),
            0,
            "a plan that writes nowhere asks for no pages, and must not ask \
             for one"
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
            unserved_in(&base, false).is_none(),
            "a plain text decode is refused, so this driver serves nothing"
        );

        // The RECURRENT refusal is the one question here about the DRIVER
        // rather than the plan, so it is asked both ways. A deployment that
        // opened a `RecurrentPool` must not have its hybrid's plan refused --
        // that plan is what the pool was allocated for -- and one that did not
        // must, emphatically: `rs_slot_ids` is where each request's carry
        // LIVES, so serving without them runs every request against slot zero.
        // One carry shared by all of them is not a fault and not a NaN.
        let hybrid = LaunchPlan {
            rs_slot_ids: vec![0, 1],
            ..base.clone()
        };
        assert!(
            unserved_in(&hybrid, false).is_some(),
            "a driver holding no slots must refuse a plan that names them"
        );
        assert!(
            unserved_in(&hybrid, true).is_none(),
            "and one that allocated them must not: the refusal was about the \
             deployment's resources, and they now exist"
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
                "`kv_len_device`",
                LaunchPlan {
                    kv_len_device: vec![0x1000],
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
            let why = unserved_in(p, false)
                .unwrap_or_else(|| panic!("{name} is ignored rather than refused"));
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
            unserved_in(&two, false).is_none(),
            "a text-only batch of two is refused for a side-channel that holds nothing"
        );
        // And the boundary is still read: a table whose total is non-zero
        // names rows, whatever its length.
        assert_eq!(
            unserved_in(
                &LaunchPlan {
                    image_indptr: vec![0, 1],
                    ..base.clone()
                },
                false
            ),
            Some("images: this driver serves text-only models"),
            "an image the plan names only in its CSR is served silently"
        );
    }

    /// The engine's own causal mask is served, and now so is every other.
    ///
    /// # What this used to assert, and why it changed
    ///
    /// `wire.rs` synthesizes `RunMask::all_true` per query row whenever it
    /// cannot elide the mask table, which is every request that is neither
    /// device-resolved nor a single-token decode. Three concurrent
    /// `chat-completion`s through a real `pie serve` came back as one success
    /// and two failures -- *this driver does not serve a user mask* -- about
    /// prompts that were entirely words.
    ///
    /// The fix then was to READ the mask and serve it when it was exactly the
    /// causal one, because a mask that would be dropped is worse than a
    /// refusal. This driver now APPLIES a mask: `requests_of` decodes each
    /// row's runs into allow-bytes, `Frame::of` packs them at the fire's
    /// widest pitch, and `attn/sdpa_paged.wgsl` reads
    /// `attention_mask[row * stride + kp]`. So the shapes this listed as
    /// refusals -- a window, a hole, a longer extent -- are answers now.
    ///
    /// The causal reading survives as an ORACLE rather than a gate: it is what
    /// says the synthesized case and no mask at all are the same computation,
    /// which is the claim the decode below has to keep.
    ///
    /// The multi-ROW request stays because `mask_indptr` is per REQUEST: a
    /// first draft read it as per-row, passed every one-row decode, and
    /// refused every prefill.
    #[test]
    fn the_engines_own_causal_mask_is_served_and_decodes_to_itself() {
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
            unserved_in(&base, false).is_none(),
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
        assert_eq!(unserved_in(&served, false), None);
        assert!(
            masks_are_exactly_causal(&served),
            "the oracle agrees this fixture is the synthesized causal mask"
        );
        // And the decode is that mask and not something near it: every key the
        // row's extent covers is allowed, and the row is exactly that long.
        let got = requests_of(&served).expect("the plan converts");
        assert_eq!(
            got.iter().map(|r| r.mask.clone()).collect::<Vec<_>>(),
            vec![
                vec![vec![1u8], vec![1, 1], vec![1, 1, 1]],
                vec![vec![1u8, 1, 1, 1, 1]],
            ],
            "a causal mask decodes to a rectangle that forbids nothing"
        );

        // A request that names NO mask is the elided case and mixes with one
        // that does. Its rows carry no mask, which is not a mask of zeros.
        let mixed = LaunchPlan {
            masks: vec![brle(5)],
            mask_indptr: vec![0, 0, 1],
            ..base.clone()
        };
        assert_eq!(unserved_in(&mixed, false), None);
        let got = requests_of(&mixed).expect("the plan converts");
        assert!(got[0].mask.is_empty(), "an elided request states no mask");
        assert_eq!(got[1].mask, vec![vec![1u8, 1, 1, 1, 1]]);

        // A NON-CANONICAL all-true encoding decodes the same. `write_skipping`
        // rebuilds a mask's runs when the trim drops KV ranges and can leave
        // zero-length falses between the pieces it kept.
        let odd = LaunchPlan {
            masks: vec![
                brle(1),
                brle(2),
                driver_api::EncodedMask::new(vec![0, 1, 0, 2], 3),
                driver_api::EncodedMask::new(vec![0, 2, 0, 3], 5),
            ],
            mask_indptr: vec![0, 3, 4],
            ..base.clone()
        };
        assert_eq!(unserved_in(&odd, false), None);
        assert_eq!(
            requests_of(&odd).expect("converts")[1].mask,
            vec![vec![1u8, 1, 1, 1, 1]],
            "all-true is a property of the mask, not of how it was written down"
        );
    }

    /// A mask that is not the causal one is APPLIED, not refused.
    ///
    /// Each of these used to be a named refusal and each is an answer now. The
    /// decoded rectangle is asserted cell by cell, because "it was accepted"
    /// says nothing about which keys survived -- and a mask read with its runs
    /// inverted, or off by one, would be accepted just as quietly.
    #[test]
    fn a_mask_that_is_not_causal_is_applied_cell_by_cell() {
        let base = LaunchPlan {
            qo_indptr: vec![0, 2],
            token_ids: vec![10, 11],
            position_ids: vec![3, 4],
            kv_len: vec![5],
            kv_page_indices: vec![0],
            kv_page_indptr: vec![0, 1],
            has_user_mask: true,
            ..LaunchPlan::default()
        };

        // A sliding window: row 0 sees keys 2..4, row 1 sees 3..5. The leading
        // FALSE run is what an inverted reading would turn into ones.
        let window = LaunchPlan {
            masks: vec![
                driver_api::EncodedMask::new(vec![2, 2], 4),
                driver_api::EncodedMask::new(vec![3, 2], 5),
            ],
            mask_indptr: vec![0, 2],
            ..base.clone()
        };
        assert_eq!(
            unserved_in(&window, false),
            None,
            "a guest mask is served now"
        );
        assert_eq!(
            requests_of(&window).expect("converts")[0].mask,
            vec![vec![0u8, 0, 1, 1], vec![0, 0, 0, 1, 1]]
        );

        // A hole: allowed, forbidden, allowed. Three runs, and the middle one
        // is the one a two-run reader would lose.
        let holed = LaunchPlan {
            masks: vec![
                driver_api::EncodedMask::new(vec![0, 1, 1, 2], 4),
                driver_api::EncodedMask::new(vec![0, 5], 5),
            ],
            mask_indptr: vec![0, 2],
            ..base
        };
        assert_eq!(unserved_in(&holed, false), None);
        assert_eq!(
            requests_of(&holed).expect("converts")[0].mask,
            vec![vec![1u8, 0, 1, 1], vec![1, 1, 1, 1, 1]]
        );
    }

    /// A mask table nobody can read is refused, and says which reading failed.
    ///
    /// The three ways a table can be unreadable, each of which would otherwise
    /// index one row's mask at another row's offset -- which is a plausible
    /// mask nobody wrote, applied to real attention.
    #[test]
    fn a_mask_table_that_does_not_close_is_refused_by_name() {
        let brle = |m: u32| driver_api::EncodedMask::new(vec![0, m], u64::from(m));
        let base = LaunchPlan {
            qo_indptr: vec![0, 3, 4],
            token_ids: vec![10, 11, 12, 13],
            position_ids: vec![0, 1, 2, 4],
            kv_len: vec![3, 5],
            kv_page_indices: vec![0, 1],
            kv_page_indptr: vec![0, 1, 2],
            ..LaunchPlan::default()
        };

        // The prefill's three rows read as one.
        let short = LaunchPlan {
            masks: vec![brle(3), brle(5)],
            mask_indptr: vec![0, 1, 2],
            ..base.clone()
        };
        assert!(
            requests_of(&short).is_err(),
            "one mask cannot stand for three rows"
        );

        // A CSR whose last boundary is past the masks it names.
        let past = LaunchPlan {
            masks: vec![brle(1), brle(2), brle(3), brle(5)],
            mask_indptr: vec![0, 3, 5],
            ..base.clone()
        };
        assert!(requests_of(&past).is_err(), "the table does not close");

        // Runs that do not cover the row. A short run list would leave the
        // tail forbidden, which is a mask nobody wrote.
        let ragged = LaunchPlan {
            masks: vec![
                brle(1),
                brle(2),
                driver_api::EncodedMask::new(vec![0, 1], 3),
                brle(5),
            ],
            mask_indptr: vec![0, 3, 4],
            ..base.clone()
        };
        assert!(
            requests_of(&ragged).is_err(),
            "a row whose runs stop short is not a mask"
        );

        // And the flag with nothing behind it is still refused at admission,
        // because there is nothing to build a rectangle from.
        assert!(
            unserved_in(
                &LaunchPlan {
                    has_user_mask: true,
                    ..base
                },
                false
            )
            .is_some_and(|s| s.contains("user mask")),
            "a mask the guest asked for with no rows on the wire"
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
    /// A request naming SEVERAL readout rows is served, and each request gets
    /// its OWN rows.
    ///
    /// This used to be a refusal, and the refusal was honest: `Step::
    /// readout_of` was one row per turn, so a caller that named `n` got one
    /// back and its program faulted four layers down. The rows always existed
    /// -- `Serving::over` forces every row to sample -- and what was missing
    /// was the mapping from a request to its row SPAN.
    ///
    /// `sample_rows_of` is that mapping's first half: the plan states readout
    /// rows GLOBALLY and `resources::Request` states them per request, so the
    /// conversion is a subtraction and a bounds check. The bounds check is the
    /// part with teeth -- a row from a NEIGHBOUR's span is a real
    /// distribution belonging to another conversation, which is the one wrong
    /// answer available here.
    #[test]
    fn a_request_reads_out_its_own_rows_however_many_it_names() {
        let base = plan(&[0, 1, 3], &[10, 11, 12], &[0, 1, 2], &[0, 1], &[0, 1, 2]);
        assert!(
            unserved_in(&base, false).is_none(),
            "a plan with no sampling table is the ordinary case and is served"
        );
        // No table: every request reads out its own last row, which
        // `Request::read` spells as an empty `samples`.
        let got = requests_of(&base).expect("converts");
        assert!(got.iter().all(|r| r.samples.is_empty()));

        // One row per request, stated.
        let one = LaunchPlan {
            sampling_indices: vec![0, 1],
            sampling_indptr: vec![0, 1, 2],
            ..base.clone()
        };
        assert_eq!(unserved_in(&one, false), None);
        let got = requests_of(&one).expect("converts");
        assert_eq!(
            got.iter().map(|r| r.samples.clone()).collect::<Vec<_>>(),
            vec![vec![0u32], vec![1]],
            "each index is already the offset within its own request"
        );

        // The second request wants two -- the speculative verifier's shape.
        let many = LaunchPlan {
            sampling_indices: vec![0, 0, 1],
            sampling_indptr: vec![0, 1, 3],
            ..base.clone()
        };
        assert_eq!(unserved_in(&many, false), None, "served now, not refused");
        let got = requests_of(&many).expect("converts");
        assert_eq!(
            got.iter().map(|r| r.samples.clone()).collect::<Vec<_>>(),
            vec![vec![0u32], vec![0, 1]],
            "both of request 1's rows, in its own numbering"
        );

        // A row past the request's own span is refused rather than
        // translated into whatever it lands on.
        let past = LaunchPlan {
            sampling_indices: vec![1, 0],
            sampling_indptr: vec![0, 1, 2],
            ..base.clone()
        };
        assert!(
            requests_of(&past).is_err(),
            "request 0 spans one row and cannot read a second"
        );

        // And a table that names NO rows for anybody asks for nothing.
        assert_eq!(
            unserved_in(
                &LaunchPlan {
                    sampling_indices: Vec::new(),
                    sampling_indptr: vec![0, 0, 0],
                    ..base
                },
                false
            ),
            None,
            "an empty sampling table asks for nothing, so there is nothing to refuse"
        );
    }
}

#[cfg(test)]
mod answered {
    use std::path::{Path, PathBuf};

    /// Every answer this crate declares, it gives.
    ///
    /// [`super::Launched::Exhausted`] was declared in this file, documented
    /// here as "evict and re-post", matched on at the engine seam in
    /// `crates/engine/src/driver/backend/wgpu.rs` -- and **constructed
    /// nowhere**. Every device that would not give the memory took the fault
    /// path instead, so a request died for a condition that clears the moment
    /// something else finishes. `Shell::admit` produces it now.
    ///
    /// A dead variant of a `pub` enum is invisible: `dead_code` does not fire
    /// on it, nothing fails to compile, and the doc beside it goes on
    /// describing an answer the caller will never see. The only thing that
    /// notices is a reader who goes looking, which is how this one was found
    /// -- after the sibling had already found and fixed its own copy.
    ///
    /// # Why every enum and not just this one
    ///
    /// Because the shape has nothing to do with `Launched`. It is available to
    /// any `pub enum` in the crate, and there are twenty-nine of them --
    /// almost all refusals, which is exactly the population where a variant
    /// nobody produces is both easy to write and impossible to notice. So the
    /// rule is asked of all of them: each variant must appear in CODE
    /// somewhere under `src/`, as `Enum::Variant`, or as `Self::Variant` in
    /// the file that declares it.
    ///
    /// It is a weak rule deliberately. It cannot tell a construction from a
    /// match -- that needs a parser -- so it catches only variants that appear
    /// NOWHERE, which is what `Exhausted` was. Reverting `Shell::admit` fails
    /// it by name, which is the calibration that matters: a sweep nobody has
    /// seen fail is a sweep that might be reading the wrong thing.
    #[test]
    fn every_answer_this_crate_declares_is_one_it_constructs() {
        /// Declared here, built somewhere else, and why that is right.
        ///
        /// Empty, and the list stays because an empty exception is a claim
        /// while a missing one is a silence.
        const ELSEWHERE: &[(&str, &str)] = &[];

        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&src, &mut files);
        assert!(files.len() > 5, "only {} sources found", files.len());

        // Declarations are read from the ORIGINAL text and constructions from
        // the blanked one: a `Display` impl is where a variant is named, never
        // where it is made.
        let scanned: Vec<(String, String)> = files
            .iter()
            .map(|(name, text)| (name.clone(), without_display(text)))
            .collect();

        let declared: Vec<(String, String, String)> = files
            .iter()
            .flat_map(|(name, text)| {
                variants(text)
                    .into_iter()
                    .map(move |(e, v)| (name.clone(), e, v))
            })
            .collect();
        assert!(
            declared.len() > 60,
            "only {} variants were parsed out of {} files, so the scan is not \
             reading the declarations it thinks it is",
            declared.len(),
            files.len()
        );

        let mut unbuilt = Vec::new();
        for (home, enum_name, variant) in &declared {
            if ELSEWHERE
                .iter()
                .any(|(n, _)| *n == format!("{enum_name}::{variant}"))
            {
                continue;
            }
            let qualified = format!("{enum_name}::{variant}");
            let shorthand = format!("Self::{variant}");
            let built = scanned.iter().any(|(name, text)| {
                text.lines().any(|line| {
                    let code = line.split_once("//").map_or(line, |(before, _)| before);
                    built(code, &qualified) || (name == home && built(code, &shorthand))
                })
            });
            if !built {
                unbuilt.push(format!("`{qualified}` (declared in {home})"));
            }
        }
        assert!(
            unbuilt.is_empty(),
            "{} variant(s) this crate declares and never BUILDS. Printing one \
             in a `Display` arm is not building it -- a caller matching on the \
             variant, and the engine seam matches on `Launched`, is reading a \
             branch that cannot be taken. Either produce it, delete it, or put \
             it in ELSEWHERE with the crate that does.\n  {}",
            unbuilt.len(),
            unbuilt.join("\n  "),
        );
    }

    /// `(enum, variant)` for every `pub enum` in one source file.
    ///
    /// A four-space indent inside a `pub enum` block, starting with a capital.
    /// Crude, and checked by the floor on the count above rather than trusted.
    fn variants(text: &str) -> Vec<(String, String)> {
        let mut found = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let Some(name) = lines[i].trim().strip_prefix("pub enum ") else {
                i += 1;
                continue;
            };
            let name = name
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .next()
                .unwrap_or("")
                .to_string();
            let mut depth = 0i32;
            for line in &lines[i..] {
                depth += line.matches('{').count() as i32;
                depth -= line.matches('}').count() as i32;
                i += 1;
                if let Some(rest) = line.strip_prefix("    ")
                    && rest.starts_with(|c: char| c.is_ascii_uppercase())
                {
                    let variant: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    let after = rest[variant.len()..].trim_start();
                    if after.starts_with('{') || after.starts_with('(') || after.starts_with(',') {
                        found.push((name.clone(), variant));
                    }
                }
                if depth == 0 {
                    break;
                }
            }
        }
        found
    }

    /// Is `needle` BUILT on this line, rather than matched on it?
    ///
    /// The distinction is what keeps this from being vacuous. Almost every
    /// enum here has an exhaustive `Display`, which names each variant as
    /// `Self::X` in a match arm -- so a rule that counted any mention would
    /// find every variant "built" by the very impl that only prints it, and
    /// `Launched` would have passed only because it happens to have no
    /// `Display`.
    ///
    /// A mention is a PATTERN when it stands to the left of a `=>` on its
    /// line, or inside a `matches!`/`if let`. Everything else is a value
    /// position, which includes the common `Foo => Self::Bar` where the
    /// variant is what the arm produces.
    ///
    /// Whole-path, too: `Failed::Wgpu` must not be satisfied by
    /// `Failed::WgpuSomething`.
    /// `text` with every `Display`/`Debug` impl body blanked out.
    ///
    /// **The loophole this closes was live and was hiding two variants.**
    /// [`built`] decides "pattern or construction" by looking for `=>` on the
    /// same LINE, so a single-line match arm is correctly read as a pattern --
    /// but a multi-line one is not:
    ///
    /// ```ignore
    /// Self::Unresolved {          // no `=>` on this line
    ///     symbol,                 // ...so it reads as a construction
    /// ```
    ///
    /// `Undispatchable::{Arity, Unresolved}` were declared, printed, and built
    /// by nothing, and this test passed over both while its own failure
    /// message said *"Printing one in a `Display` arm is not building it"*.
    ///
    /// Blanking the impl is coarser than parsing and is the right coarseness:
    /// a `Display` body's whole job is to name every variant, so nothing in
    /// one can ever be evidence that a variant is CONSTRUCTED.
    fn without_display(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut lines = text.lines().peekable();
        while let Some(line) = lines.next() {
            let opens = line.trim_start().starts_with("impl ")
                && (line.contains("fmt::Display for") || line.contains("fmt::Debug for"));
            out.push_str(line);
            out.push('\n');
            if !opens {
                continue;
            }
            // Skip to the impl's closing brace, counting from this line so a
            // `{` on the `impl` line itself is the one being matched.
            let mut depth = line.matches('{').count() - line.matches('}').count();
            while depth > 0 {
                let Some(inner) = lines.next() else { break };
                depth += inner.matches('{').count();
                depth -= inner.matches('}').count();
                out.push('\n');
            }
        }
        out
    }

    fn built(code: &str, needle: &str) -> bool {
        let arrow = code.find("=>");
        let guarded = code.contains("matches!(") || code.contains("if let ");
        let mut from = 0;
        while let Some(at) = code[from..].find(needle) {
            let at = from + at;
            from = at + 1;
            let after = code[at + needle.len()..].chars().next();
            if after.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            let pattern = guarded || arrow.is_some_and(|a| at < a);
            if !pattern {
                return true;
            }
        }
        false
    }

    fn walk(dir: &Path, into: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, into);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Ok(text) = std::fs::read_to_string(&path) {
                    into.push((name, text));
                }
            }
        }
    }
}
