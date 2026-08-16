//! Filling in the geometry a step does not carry, and translating the pages
//! it does.
//!
//! # Two jobs, one pass, because they are the same arithmetic
//!
//! A step arrives with its members partitioned into geometry classes. A
//! `Host` member states its whole fire on the wire; a `DecodeEnvelope` member
//! states almost none of it, because the numbers -- which token, at which
//! position, over how much cache -- were computed by the PREVIOUS fire's
//! program and put on channels, and never came to the host at all. What both
//! kinds share is that the pages they name are WORKING-SET pages: page 0 of
//! this conversation, not page 0 of the pool.
//!
//! # The pages, and the bug that is invisible with one conversation
//!
//! `LaunchPlan::kv_page_indices` are indices into the request's own working
//! set, and the frame's `kv_translation` says which physical page each one
//! is. The engine checks a fire's pages against the working set's readable
//! DECLARATION -- `page 7 escapes the readable declaration` -- which is a
//! statement about logical pages, and it sizes the pool from the
//! TRANSLATION's maximum, which is a statement about physical ones.
//!
//! A driver that skips the translation reads page 0 of the pool for every
//! conversation. With one conversation in flight that is accidentally right,
//! which is exactly why it survives a single-request test: every fire agrees
//! with every other fire about a mapping that happens to be consistent. With
//! two, the second conversation attends the first one's keys and answers
//! fluently in the first one's context.
//!
//! # What a decode envelope may and may not be told
//!
//! The class advertises three ports -- `EmbedTokens`, `Positions`, `KvLen` --
//! and those three are read from the instance's channels through
//! [`driver::resolve`], the backend-neutral copier. The PAGES are not a port
//! of this class: the host still owns page placement, so the pages a decode
//! attends are its working set's first `ceil(len / page)` pages, translated
//! like any other. That is the whole of the difference, and it is why a
//! decode step's wire tables are empty rather than wrong.

use driver_api::{FrameSubmission, LaunchPlan, StepSubmission};

use crate::frames::{Unlaunched, member_requests};
use crate::programs::Programs;

/// What filling a step's geometry produced.
#[derive(Debug)]
pub enum Filled {
    /// The step can be fired, with this plan in place of the wire's.
    ///
    /// `writes` is the STATED write descriptor -- one entry per row of
    /// `plan`, `None` where that row's member stated none -- and is empty
    /// when no member stated one at all, which is every host and envelope
    /// fire. PER ROW and not per step, because a step MIXES: a
    /// device-geometry member co-batches with host members, and the first
    /// cut of this refused such a step ("some members bind a write
    /// descriptor and some do not") rather than serving both. See
    /// `Shell::prepare`.
    Ready {
        /// The geometry this step fires over.
        plan: Box<LaunchPlan>,
        /// Where each row's KV goes, when the program said so.
        writes: Vec<Option<(u32, u32)>>,
    },
    /// A descriptor channel is empty: the program that fills it has not run.
    ///
    /// Not a failure. The producer is usually the step BEFORE this one in the
    /// same frame, so the remedy is to fire that one first; when it is a
    /// program in another frame the scheduler re-posts. Either way nothing of
    /// this step has happened, which is what makes re-posting safe.
    Early {
        /// The channel with nothing in it.
        channel: u32,
    },
}

/// The geometry class of one member of a step, or `None` when the step's own
/// tables do not say.
///
/// # Why not-said is not `Host`
///
/// A member outside every sub-batch window used to read back as `Host`, which
/// is a class this driver knows how to serve -- so the guess did not stop
/// anything, it just sent the member down a path that reads geometry out of
/// the WIRE PLAN. A device-geometry member states its pages in a channel and
/// leaves the wire plan's tables empty, so the guess surfaced four steps later
/// as `request 0 has no page span in a CSR of 0 entries`: true, useless, and
/// blaming the plan for a decision made here.
///
/// An empty `sub_batch_indptr` is different and stays `Host`: [`validate`]
/// admits an absent table (`allow_empty`), and a step with no sub-batching has
/// no other class it could be. What is refused is a table that EXISTS and does
/// not cover this member, or one whose class list is shorter than its own CSR
/// -- both of which mean the step said something about its members and left
/// this one out.
///
/// [`validate`]: driver_api::StepSubmission::validate
fn class_of(sub: &StepSubmission, member: usize) -> Option<u32> {
    if sub.sub_batch_indptr.is_empty() {
        return Some(driver_api::PIE_GEOMETRY_CLASS_HOST);
    }
    for (b, window) in sub.sub_batch_indptr.windows(2).enumerate() {
        if (window[0] as usize..window[1] as usize).contains(&member) {
            return sub.sub_batch_class.get(b).copied();
        }
    }
    None
}

/// The physical pages of one roster row, in working-set order.
///
/// # Why an empty answer is not one answer
///
/// [`physical`] reads an empty segment as the IDENTITY: no placement was made,
/// so a fire's logical page number is its physical one. That is right for a
/// frame that placed nothing, and it is the case
/// `a_frame_that_places_no_pages_leaves_them_alone` measures.
///
/// It is not right for a frame that HAS a translation table which does not
/// describe this row -- a CSR whose boundaries cross, or whose end runs past
/// the table it indexes. Answering `&[]` there turns a malformed table into an
/// identity translation, and an identity translation on a paged pool is the
/// failure this whole crate is written against: the fire reads whatever page
/// carries that logical number, which is some other conversation's, and
/// answers fluently.
///
/// So the two are separated. No table is `Ok(&[])`; a table that cannot speak
/// for this row is a refusal naming the row and the numbers that did not
/// close.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] for a translation CSR that does not describe
/// `row`.
/// The rows request `r` names in the wire plan's sampling table.
///
/// Empty for a plan with no table, which means every request reads out its own
/// last row -- the decode case. Empty too for a CSR that does not name this
/// request, which `validate_geometry` has already had its say about; answering
/// with another request's span would be the one wrong answer available here.
fn sampling_rows(plan: &driver_api::LaunchPlan, r: usize) -> &[u32] {
    let (Some(&lo), Some(&hi)) = (plan.sampling_indptr.get(r), plan.sampling_indptr.get(r + 1))
    else {
        return &[];
    };
    if hi < lo {
        return &[];
    }
    plan.sampling_indices
        .get(lo as usize..hi as usize)
        .unwrap_or(&[])
}

/// One row of allow-bytes as the alternating runs the wire spells.
///
/// The encoding starts with a FALSE run and alternates, which is the reading
/// `frames::decode_mask` and `frames::all_true` both use -- an even index is a
/// forbidden run, an odd one allowed. A row that begins allowed therefore
/// opens with a zero-length false run, and that is not a special case: it is
/// what `write_skipping` already produces when a trim drops a leading range.
///
/// The round trip is what makes this safe to do at all: the driver resolves a
/// DENSE mask and every other path in this crate reads runs, so re-encoding
/// here is what lets one reader serve both instead of two readers agreeing by
/// inspection.
fn runs_of(row: &[u8]) -> driver_api::EncodedMask {
    let mut runs: Vec<u32> = Vec::new();
    let mut want = 0u8;
    let mut at = 0usize;
    while at < row.len() {
        let mut n = 0u32;
        while at < row.len() && u8::from(row[at] != 0) == want {
            n += 1;
            at += 1;
        }
        runs.push(n);
        want ^= 1;
    }
    driver_api::EncodedMask::new(runs, row.len() as u64)
}

fn translation(frame: &FrameSubmission, row: u32) -> Result<&[u32], Unlaunched> {
    // No table at all: nothing was placed, and `physical` reads that as the
    // identity on purpose.
    if frame.kv_translation_indptr.is_empty() {
        return Ok(&[]);
    }
    let at = row as usize;
    match (
        frame.kv_translation_indptr.get(at),
        frame.kv_translation_indptr.get(at + 1),
    ) {
        (Some(&lo), Some(&hi)) if hi >= lo && hi as usize <= frame.kv_translation.len() => {
            Ok(&frame.kv_translation[lo as usize..hi as usize])
        }
        (Some(&lo), Some(&hi)) => Err(Unlaunched::Malformed(format!(
            "this frame's translation of row {row} is {lo}..{hi} of a table \
             holding {} page(s)",
            frame.kv_translation.len()
        ))),
        _ => Err(Unlaunched::Malformed(format!(
            "this frame's translation names {} boundaries, which do not \
             describe row {row}",
            frame.kv_translation_indptr.len()
        ))),
    }
}

/// One working-set page as the physical page it is placed in.
///
/// An EMPTY translation means the frame states none, which is what this
/// driver's own tests and the worker's single-request harness build, and the
/// only honest reading of a frame that names no placement is that the pages
/// it names are already physical.
fn physical(segment: &[u32], logical: u32) -> Result<u32, Unlaunched> {
    if segment.is_empty() {
        return Ok(logical);
    }
    segment
        .get(logical as usize)
        .copied()
        .filter(|&page| page != u32::MAX)
        .ok_or_else(|| {
            Unlaunched::Malformed(format!(
                "this fire names working-set page {logical}, which the frame's translation of \
                 {} page(s) does not place",
                segment.len()
            ))
        })
}

/// How many pages a sequence of `len` tokens occupies.
fn pages_for(len: u32, page: u32) -> u32 {
    if page == 0 { 0 } else { len.div_ceil(page) }
}

/// This step's plan with every member's geometry known and every page
/// physical.
///
/// # Errors
///
/// [`Unlaunched::Malformed`] for a CSR that does not close, a page the
/// frame's translation does not place, or a descriptor channel whose cell is
/// not a geometry index.
pub fn fill(
    programs: &Programs,
    frame: &FrameSubmission,
    sub: &StepSubmission,
    page: u32,
) -> Result<Filled, Unlaunched> {
    let wire_rows = sub.plan.qo_indptr.len().saturating_sub(1);
    let mut out = sub.plan.clone();
    let (mut tokens, mut positions, mut qo) = (Vec::new(), Vec::new(), vec![0u32]);
    let (mut pages, mut page_indptr, mut lens) = (Vec::new(), vec![0u32], Vec::new());
    // The DENSE mask a device-geometry program resolves, re-encoded as the
    // runs the rest of this driver reads. Two vectors rather than one because
    // `mask_indptr` is per REQUEST and the runs are per ROW.
    let (mut masks, mut mask_indptr) = (Vec::new(), vec![0u32]);
    let mut any_mask = false;
    // The read-out rows, REMAPPED. `fill` rebuilds the row layout, so a wire
    // row index means nothing here -- but the offset does: request `r`'s rows
    // start at `base` in the new layout and at `lo` in the old, so row `x`
    // becomes `base + (x - lo)`.
    let (mut samples, mut sample_indptr) = (Vec::new(), vec![0u32]);
    // The STATED write descriptor, one entry per row, and empty unless a
    // member states one. `W_SLOT`/`W_OFF` are ports this seam claims, and
    // until they were read this driver wrote where the POSITION said instead
    // -- the same cell for two lanes of a beam that had forked.
    let mut writes: Vec<Option<(u32, u32)>> = Vec::new();

    for (member, &row) in sub.roster_rows.iter().enumerate() {
        let segment = translation(frame, row)?;
        let class = class_of(sub, member).ok_or_else(|| {
            Unlaunched::Malformed(format!(
                "step member {member} is in no sub-batch: {} boundaries covering members \
                 0..{}, and {} class(es)",
                sub.sub_batch_indptr.len(),
                sub.sub_batch_indptr.last().copied().unwrap_or(0),
                sub.sub_batch_class.len()
            ))
        })?;
        // Both device-resolved classes are answered from the SAME resolution
        // and differ in one thing: where a request's pages come from.
        //
        // `DecodeEnvelope` binds three ports -- tokens, positions, kv_len --
        // and the paging is the driver's own arithmetic over the frame's
        // translation, because a decode's pages are its history and its
        // position says how much of that there is. `DeviceGeometry` binds
        // seven: the program traces its whole geometry in-graph, so the pages,
        // the CSR and the write descriptor are READ rather than derived.
        //
        // Keeping them one branch is what stops the two from drifting on the
        // five things they agree about.
        if class == driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE
            || class == driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY
        {
            let id = *frame.instance_ids.get(row as usize).ok_or_else(|| {
                Unlaunched::Malformed(format!(
                    "step member {member} names roster row {row} of {}",
                    frame.instance_ids.len()
                ))
            })?;
            let resolved = match programs
                .geometry(id, page)
                .map_err(|e| Unlaunched::Malformed(format!("{e}")))?
            {
                driver::Resolution::Ready(geometry) => geometry,
                driver::Resolution::NotReady { channel } => {
                    return Ok(Filled::Early { channel });
                }
                driver::Resolution::Failed { message } => {
                    return Err(Unlaunched::Malformed(format!(
                        "instance {id} resolves no geometry: {message}"
                    )));
                }
            };
            for (which, request) in resolved.qo_indptr.windows(2).enumerate() {
                let (lo, hi) = (request[0] as usize, request[1] as usize);
                if lo > hi || hi > resolved.token_ids.len() || hi > resolved.position_ids.len() {
                    return Err(Unlaunched::Malformed(format!(
                        "instance {id} resolves a request over rows {lo}..{hi} of {} token(s)",
                        resolved.token_ids.len()
                    )));
                }
                tokens.extend_from_slice(&resolved.token_ids[lo..hi]);
                positions.extend_from_slice(&resolved.position_ids[lo..hi]);
                qo.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
                // The WRITE DESCRIPTOR, honoured by being READ.
                //
                // This seam claims `PIE_DEVICE_GEOMETRY_PORTS`, which names
                // `W_SLOT` and `W_OFF`: where each row's KV goes, stated by
                // the program rather than worked out from its position. This
                // driver used to ignore them and append at
                // `pages[position / page_size]`, which agrees with the
                // program for every fire whose rows append at their own
                // positions -- and disagrees for a BEAM, where the lanes fork
                // and `w_slot` is how a lane says which copy of a page is its
                // own. Two lanes then wrote one cell, which `Frame::of` saw
                // as `SharedPage` and refused: this driver noticing its own
                // wrong plan rather than the plan being wrong.
                //
                // Translated like every other page this class states: the
                // program numbers pages within its WORKING SET and the frame
                // says which physical page each one is.
                if resolved.has_write_desc {
                    for row in lo..hi {
                        let (Some(&logical), Some(&off)) =
                            (resolved.w_page.get(row), resolved.w_off.get(row))
                        else {
                            return Err(Unlaunched::Malformed(format!(
                                "instance {id} binds a write descriptor and states {} slot(s) \
                                 and {} offset(s) for {} row(s)",
                                resolved.w_page.len(),
                                resolved.w_off.len(),
                                resolved.position_ids.len()
                            )));
                        };
                        if off >= page {
                            return Err(Unlaunched::Malformed(format!(
                                "instance {id} states that row {row} writes offset {off} of a \
                                 {page}-token page"
                            )));
                        }
                        writes.push(Some((physical(segment, logical)?, off)));
                    }
                } else {
                    writes.extend(std::iter::repeat_n(None, hi - lo));
                }
                let last = resolved.position_ids[lo..hi].iter().copied().max();
                // A device-geometry program STATES its pages, so they are read
                // off the resolution and translated. `has_kv_family` is what
                // says the family was bound at all; a program of this class
                // that bound none is refused rather than quietly given the
                // envelope's arithmetic, because the two answer differently
                // the moment a guest's pages are not its own history in order
                // -- which is the whole reason to trace them.
                let stated_pages = class == driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY;
                if stated_pages && !resolved.has_kv_family {
                    return Err(Unlaunched::Malformed(format!(
                        "instance {id} is device-geometry and binds no KV family: its \
                         pages would have to be guessed from its positions, which is a \
                         different fire from the one it traced"
                    )));
                }
                if stated_pages {
                    let (plo, phi) = match (
                        resolved.kv_page_indptr.get(which),
                        resolved.kv_page_indptr.get(which + 1),
                    ) {
                        (Some(&lo), Some(&hi)) => (lo as usize, hi as usize),
                        _ => {
                            return Err(Unlaunched::Malformed(format!(
                                "instance {id} resolves {} page boundaries for {} request(s)",
                                resolved.kv_page_indptr.len(),
                                resolved.qo_indptr.len().saturating_sub(1)
                            )));
                        }
                    };
                    if plo > phi || phi > resolved.kv_page_indices.len() {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} resolves request {which} over pages {plo}..{phi} \
                             of {}",
                            resolved.kv_page_indices.len()
                        )));
                    }
                    // Through the same translation every other path uses: the
                    // program states LOGICAL pages and the frame says which
                    // physical page each one is. A logical number used as a
                    // physical one is another conversation's pages.
                    for &logical in &resolved.kv_page_indices[plo..phi] {
                        pages.push(physical(segment, logical)?);
                    }
                } else {
                    // The pages a decode attends are its own history, and the
                    // last position it writes says how much of that there is.
                    // Taken from the POSITIONS rather than from `kv_len`,
                    // because a row writes where its position says and reading
                    // one page fewer than it writes is an attention over a page
                    // this fire itself filled.
                    let live = last.map_or(0, |p| pages_for(p.saturating_add(1), page));
                    for logical in 0..live {
                        pages.push(physical(segment, logical)?);
                    }
                }
                page_indptr.push(u32::try_from(pages.len()).unwrap_or(u32::MAX));
                let derived = last.map_or(0, |p| driver::last_page_len(p.saturating_add(1), page));
                // The `kv_len` PORT, honoured by being checked.
                //
                // This seam claims `PIE_DECODE_ENVELOPE_PORTS`, which names
                // `EMBED_TOKENS`, `POSITIONS` and `KV_LEN`. The first two are
                // read straight off the resolution above. The third is
                // DERIVED here, from the position, for the reason the comment
                // above gives -- and a derivation is not a reading. Where the
                // two agree, nothing is lost and this costs a comparison.
                // Where they disagree, the guest asked for a history of one
                // length and this fire would attend another, which is a
                // silent answer to a question it was asked out loud.
                //
                // They agree for every decode the engine builds today: a
                // guest's `kv_len.put(&next_length)` and its
                // `positions.put(&length)` come off the same counter. A guest
                // that parted them -- a sliding window, an evicted cache --
                // gets a refusal naming both numbers instead of an attention
                // over a span it did not ask for.
                if stated_pages {
                    // A device-geometry program states its own span, and its
                    // pages are not required to be its history in order -- an
                    // evicted cache or a sliding window is exactly what this
                    // class exists to express -- so the derivation above is
                    // not a second opinion here, it is a different question.
                    // The STATED value is the answer, and its absence is a
                    // refusal rather than a fallback to the derivation.
                    let Some(&stated) = resolved.kv_last_page_lens.get(which) else {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} is device-geometry and states no last-page \
                             length for request {which}"
                        )));
                    };
                    if stated > page {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} states a last page of {stated} rows in a pool \
                             whose pages hold {page}"
                        )));
                    }
                    lens.push(stated);
                    // The dense mask this program resolved, one byte per
                    // lane, cut into the rows of THIS request and re-encoded
                    // as runs. `mask_key_len` is the row pitch the resolver
                    // derived; a row is `[j * pitch, j * pitch + pitch)`.
                    if resolved.has_mask {
                        let pitch = resolved.mask_key_len as usize;
                        if pitch == 0 {
                            return Err(Unlaunched::Malformed(format!(
                                "instance {id} resolves a mask with no key extent"
                            )));
                        }
                        for j in lo..hi {
                            let from = j * pitch;
                            let to = from + pitch;
                            if to > resolved.mask.len() {
                                return Err(Unlaunched::Malformed(format!(
                                    "instance {id} resolves row {j} of a {pitch}-wide mask \
                                     from {} byte(s)",
                                    resolved.mask.len()
                                )));
                            }
                            masks.push(runs_of(&resolved.mask[from..to]));
                        }
                        any_mask = true;
                    }
                    mask_indptr.push(u32::try_from(masks.len()).unwrap_or(u32::MAX));
                    // The rows this program asked to read, which it states in
                    // a channel like everything else of this class.
                    //
                    // Dropping them was a real defect and a quiet one:
                    // `cacheback-speculative-decoding`'s verifier names one
                    // row per drafted token, and a fall-through to "the last
                    // row" answers a five-row question with one row. The
                    // guest's own length check does not catch it -- the
                    // channel's shape is what it measures -- so it verified
                    // drafts against a distribution belonging to the wrong
                    // position and diverged from its sequential control at
                    // the first rejection.
                    //
                    // Numbered from the REQUEST: index 0 is this request's
                    // first row. Measured twice on the scheduler's own output
                    // -- `qo=[0, 92, 93] sidx=[91, 0]`, where request 1 spans
                    // rows 92..93 and names `0`, its own -- and
                    // `driver::resolve` pushes `span - 1` from the request's
                    // own row count when no read-out port is bound. See the
                    // field's doc in `driver-api`, which also records the
                    // branch that is NOT settled: a guest-bound
                    // `Port::Readout` copies the guest's values through
                    // unchecked, and nothing pins what they mean.
                    let named = resolved
                        .sampling_indptr
                        .get(which)
                        .zip(resolved.sampling_indptr.get(which + 1))
                        .map(|(&a, &b)| (a as usize, b as usize));
                    if let Some((rlo, rhi)) = named {
                        if rlo > rhi || rhi > resolved.sampling_indices.len() {
                            return Err(Unlaunched::Malformed(format!(
                                "instance {id} spans readouts {rlo}..{rhi} of {}",
                                resolved.sampling_indices.len()
                            )));
                        }
                        for &row in &resolved.sampling_indices[rlo..rhi] {
                            let row = row as usize;
                            if row >= hi - lo {
                                return Err(Unlaunched::Malformed(format!(
                                    "instance {id} reads out row {row} of request {which}, \
                                     which has {} row(s)",
                                    hi - lo
                                )));
                            }
                            samples.push(u32::try_from(row).unwrap_or(u32::MAX));
                        }
                    }
                    sample_indptr.push(u32::try_from(samples.len()).unwrap_or(u32::MAX));
                } else {
                    if let Some(&stated) = resolved.kv_last_page_lens.get(which)
                        && stated != derived
                    {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} resolves request {which} with a last page of \
                             {stated}, and its own positions say {derived}"
                        )));
                    }
                    lens.push(derived);
                    // No mask on this class, and the CSR still needs a
                    // boundary per request: a table one short would give the
                    // next request this one's runs.
                    mask_indptr.push(u32::try_from(masks.len()).unwrap_or(u32::MAX));
                    // Likewise for the read-out: a device-resolved class names
                    // no rows, which `Request::read` takes as "the last one".
                    sample_indptr.push(u32::try_from(samples.len()).unwrap_or(u32::MAX));
                }
            }
            continue;
        }

        let (first, last) = member_requests(&sub.program_row_indptr, member, wire_rows)
            .ok_or_else(|| {
                Unlaunched::Malformed(format!(
                    "step member {member} is not described by the {}-entry attribution CSR \
                     over {wire_rows} request(s)",
                    sub.program_row_indptr.len()
                ))
            })?;
        for r in first..last {
            let (lo, hi) = (
                sub.plan.qo_indptr[r] as usize,
                sub.plan.qo_indptr[r + 1] as usize,
            );
            if lo > hi || hi > sub.plan.token_ids.len() || hi > sub.plan.position_ids.len() {
                return Err(Unlaunched::Malformed(format!(
                    "request {r} spans rows {lo}..{hi} of {} token(s)",
                    sub.plan.token_ids.len()
                )));
            }
            tokens.extend_from_slice(&sub.plan.token_ids[lo..hi]);
            positions.extend_from_slice(&sub.plan.position_ids[lo..hi]);
            // A wire member states no write target: this driver derives it,
            // which is what it has always done and what `None` asks for.
            writes.extend(std::iter::repeat_n(None, hi - lo));
            qo.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
            // Numbered from the REQUEST, same as the resolved path above. A
            // row past this request's own count would read a real
            // distribution belonging to another position, so it is refused
            // here rather than translated into whatever it lands on.
            for &row in sampling_rows(&sub.plan, r) {
                let row = row as usize;
                if row >= hi - lo {
                    return Err(Unlaunched::Malformed(format!(
                        "request {r} reads out row {row}, and has {} row(s)",
                        hi - lo
                    )));
                }
                samples.push(u32::try_from(row).unwrap_or(u32::MAX));
            }
            sample_indptr.push(u32::try_from(samples.len()).unwrap_or(u32::MAX));
            let (plo, phi) = match (
                sub.plan.kv_page_indptr.get(r),
                sub.plan.kv_page_indptr.get(r + 1),
            ) {
                (Some(&plo), Some(&phi)) if phi >= plo => (plo as usize, phi as usize),
                _ => {
                    // A `Host` member with no page table cannot attend
                    // anything, and the wire contract permits the table to be
                    // absent (`csr(.., allow_empty = true)`) because the two
                    // DEVICE classes state their pages in channels instead.
                    // So an empty CSR here is not a malformed plan on its own
                    // -- it is a member sent to the host path carrying no host
                    // geometry, which is what the class is named for.
                    //
                    // Seen for real: `crates/engine/src/pipeline/fire.rs` sets
                    // `device_resolved_geometry` from `decode_envelope`
                    // alone, so a POOLED device-geometry fire (the engine logs
                    // "executes as a pool-owned device-geometry pass" for it)
                    // reaches `scheduler::batch`, which stamps only `Host` or
                    // `DecodeEnvelope`, as class 0 with its geometry still in
                    // the channels. Naming the class is what makes that one
                    // step to find rather than four.
                    return Err(Unlaunched::Malformed(format!(
                        "request {r} has no page span in a CSR of {} entries, and this \
                         member is geometry class {class}: a host-class member states its \
                         own pages, and only classes {} and {} state them elsewhere",
                        sub.plan.kv_page_indptr.len(),
                        driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                        driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY
                    )));
                }
            };
            if phi > sub.plan.kv_page_indices.len() {
                return Err(Unlaunched::Malformed(format!(
                    "request {r} spans pages {plo}..{phi} of {} page indices",
                    sub.plan.kv_page_indices.len()
                )));
            }
            for &logical in &sub.plan.kv_page_indices[plo..phi] {
                pages.push(physical(segment, logical)?);
            }
            page_indptr.push(u32::try_from(pages.len()).unwrap_or(u32::MAX));
            lens.push(sub.plan.kv_last_page_lens.get(r).copied().unwrap_or(0));
        }
    }

    out.token_ids = tokens;
    out.position_ids = positions;
    if any_mask {
        out.masks = masks;
        out.mask_indptr = mask_indptr;
        out.has_user_mask = true;
        // The FLAG goes off: it means "the driver resolves a dense per-cell
        // mask pre-forward", and by here it has. What is on the plan now is
        // the same runs a host-lowered mask would have carried, which is what
        // `frames::requests_of` reads.
        out.dense_device_mask = false;
    }
    out.qo_indptr = qo;
    out.kv_page_indices = pages;
    out.kv_page_indptr = page_indptr;
    out.kv_last_page_lens = lens;
    // REMAPPED, not dropped. A sampling table indexed against the wire's rows
    // would name rows this plan no longer has -- `fill` rebuilds the layout --
    // and dropping it lost the one thing a speculative verifier needs: which
    // rows of its own it asked to read. The remap above is the offset, done
    // where both layouts are in hand.
    out.sampling_indices = samples;
    out.sampling_indptr = sample_indptr;
    // Row-aligned, and dropped whole when nothing stated one: an entry per
    // row is what lets a MIXED step be served, and an empty table is what
    // says "derive everything", which is every fire but a device-geometry
    // one.
    //
    // This one is an INVARIANT on `fill`'s own construction, not a refusal of
    // anything a guest can state, and it is unreachable today: both branches
    // above extend `tokens` and `writes` by the same `hi - lo`. It is here
    // because the next path that produces rows will be written by somebody
    // who does not know that, and a table that is short by one misaligns
    // EVERY row after it -- silently, since `Frame::of` would still find an
    // entry for each row it asks about. Caught here, it names itself.
    if writes.len() != out.token_ids.len() {
        return Err(Unlaunched::Malformed(format!(
            "this step built {} write slot(s) for {} row(s)",
            writes.len(),
            out.token_ids.len()
        )));
    }
    if writes.iter().all(Option::is_none) {
        writes.clear();
    }
    Ok(Filled::Ready {
        plan: Box::new(out),
        writes,
    })
}

#[cfg(test)]
mod tests {
    /// A dense row and its runs say the same thing, in both directions.
    ///
    /// The round trip is the whole safety of resolving a DENSE mask into a
    /// driver whose every other path reads RUNS. Checked against
    /// `frames::decode_mask`, which is the reader that will consume these --
    /// not against a second encoder written here, which would only be
    /// checking this one against itself.
    #[test]
    fn a_dense_mask_row_and_its_runs_are_the_same_mask() {
        for row in [
            vec![1u8, 1, 1],
            vec![0u8, 0, 0],
            vec![0u8, 1, 1, 0, 1],
            vec![1u8, 0, 1],
            vec![1u8],
            vec![0u8],
            // Non-zero is "allowed", not "one".
            vec![7u8, 0, 3],
        ] {
            let runs = super::runs_of(&row);
            assert_eq!(runs.total_size, row.len() as u64);
            let back = crate::frames::decode_mask_for_test(&runs)
                .unwrap_or_else(|e| panic!("{row:?} does not decode: {e:?}"));
            let want: Vec<u8> = row.iter().map(|b| u8::from(*b != 0)).collect();
            assert_eq!(back, want, "{row:?} did not survive the round trip");
        }
    }

    /// An all-allowed row opens with a zero-length forbidden run.
    ///
    /// Stated because it looks like a bug and is the encoding: an even index
    /// is a FORBIDDEN run, so a row that begins allowed has to say "none
    /// forbidden" first. `write_skipping` produces the same shape when a trim
    /// drops a leading range, and `frames`' own tests pin that such a mask is
    /// read as all-true.
    #[test]
    fn a_row_that_begins_allowed_opens_with_an_empty_run() {
        assert_eq!(super::runs_of(&[1, 1, 1]).runs, vec![0, 3]);
        assert_eq!(super::runs_of(&[0, 1, 1]).runs, vec![1, 2]);
    }

    use driver_api::plan::{LaunchChannel, LaunchPackage, LaunchPort};

    use super::*;

    /// A frame of one roster row placing four working-set pages.
    fn frame(translation: Vec<u32>, sub: StepSubmission) -> FrameSubmission {
        FrameSubmission {
            instance_ids: vec![7],
            kv_translation_indptr: vec![0, u32::try_from(translation.len()).unwrap()],
            kv_translation: translation,
            required_kv_pages: 4,
            steps: vec![sub],
        }
    }

    /// One host-wire step: two requests, each naming its own pages.
    fn wire() -> StepSubmission {
        StepSubmission {
            plan: LaunchPlan {
                token_ids: vec![11, 12, 13],
                position_ids: vec![0, 1, 17],
                kv_page_indices: vec![0, 1, 0, 1],
                kv_page_indptr: vec![0, 2, 4],
                kv_last_page_lens: vec![2, 2],
                qo_indptr: vec![0, 2, 3],
                ..LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_HOST],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 2],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: Vec::new(),
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }
    }

    /// The pages a fire names are its WORKING SET's, and the frame says where
    /// each one is.
    ///
    /// # The bug this pins
    ///
    /// Every conversation's first page is page 0 of its own working set, and
    /// they are not the same page of the pool. A driver that skips this
    /// translation gives them all pool page 0: the second conversation reads
    /// the first one's keys and answers fluently in its context, and no
    /// single-request test can see it because one conversation's identity map
    /// is consistent with itself.
    #[test]
    fn a_fires_pages_are_translated_to_the_ones_the_frame_placed() {
        let sub = wire();
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let Filled::Ready { plan, .. } =
            fill(&Programs::new(), &frame, &frame.steps[0], 16).expect("a well-formed step")
        else {
            panic!("a host-wire step is never early: it names all of its own geometry");
        };
        assert_eq!(
            plan.kv_page_indices,
            vec![3, 2, 3, 2],
            "working-set pages 0 and 1 are physical pages 3 and 2"
        );
        assert_eq!(plan.kv_page_indptr, sub.plan.kv_page_indptr);
        assert_eq!(plan.token_ids, sub.plan.token_ids);
        assert_eq!(plan.qo_indptr, sub.plan.qo_indptr);
    }

    /// A step that says something about its members and leaves one out is
    /// refused, because the alternative is serving it as a class nobody chose.
    #[test]
    fn a_member_in_no_sub_batch_is_refused_rather_than_called_host() {
        let mut sub = wire();
        // A table that EXISTS and covers nothing: the step has a sub-batch
        // list and member 0 is not in it.
        sub.sub_batch_indptr = vec![0, 0];
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let error = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect_err("a member in no sub-batch has no class to be served as");
        let said = format!("{error}");
        assert!(
            said.contains("step member 0 is in no sub-batch"),
            "the refusal names the member: {said}"
        );
    }

    /// The same hole from the other side: a window exists, and the class list
    /// is too short to say what is in it.
    #[test]
    fn a_sub_batch_with_no_class_is_refused_rather_than_called_host() {
        let mut sub = wire();
        sub.sub_batch_class = Vec::new();
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let error = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect_err("a sub-batch with no class states no class");
        assert!(format!("{error}").contains("is in no sub-batch"), "{error}");
    }

    /// An ABSENT table still reads as `Host`, which is the wire contract:
    /// `StepSubmission::validate` admits an empty `sub_batch_indptr`, and a
    /// step with no sub-batching has no other class it could be.
    #[test]
    fn a_step_with_no_sub_batch_table_at_all_is_host() {
        let mut sub = wire();
        sub.sub_batch_indptr = Vec::new();
        sub.sub_batch_class = Vec::new();
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let Filled::Ready { plan, .. } = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect("an absent sub-batch table is a host step")
        else {
            panic!("a host-wire step is never early");
        };
        assert_eq!(plan.kv_page_indices, vec![3, 2, 3, 2]);
    }

    /// A host member with no page table is refused in terms of its CLASS,
    /// because the empty table is wire-legal and the class is what makes it
    /// wrong.
    #[test]
    fn a_host_member_with_no_page_table_is_refused_naming_its_class() {
        let mut sub = wire();
        sub.plan.kv_page_indptr = Vec::new();
        sub.plan.kv_page_indices = Vec::new();
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let error = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect_err("a host member states its own pages");
        let said = format!("{error}");
        assert!(
            said.contains("geometry class 0") && said.contains("state them elsewhere"),
            "the refusal names the class and what the other classes do: {said}"
        );
    }

    /// What `fill` writes is what `frames` reads: the CONTRACT between them,
    /// which neither module's own tests could see.
    ///
    /// # The bug this pins
    ///
    /// `fill` lays the fire out afresh and states its read-out rows in the
    /// layout it built (`base + row`). `frames::requests_of` then cuts them
    /// back into per-request offsets. Each module had tests, both passed, and
    /// the two disagreed about the numbering for a while: reading `fill`'s
    /// output as if it were the SCHEDULER's -- request-relative -- refuses
    /// every request of a member but the first, because its rows do not start
    /// at zero. Every single-request plan agrees under both readings, which is
    /// why only a two-request member shows it.
    #[test]
    fn what_fill_writes_is_what_frames_reads() {
        let mut sub = wire();
        // `wire()` is two requests: rows 0..2 and 2..3. Each reads its own
        // last row, stated in the numbering the scheduler uses.
        sub.plan.sampling_indices = vec![1, 0];
        sub.plan.sampling_indptr = vec![0, 1, 2];
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let Filled::Ready { plan, .. } =
            fill(&Programs::default(), &frame, &frame.steps[0], 16).expect("a well-formed step")
        else {
            panic!("a host-wire step is never early");
        };
        let requests = crate::frames::requests_of(&plan).expect("what fill wrote is servable");
        assert_eq!(
            requests
                .iter()
                .map(|r| r.samples.clone())
                .collect::<Vec<_>>(),
            vec![vec![1u32], vec![0]],
            "each request reads its own last row, back in its own numbering"
        );
    }

    /// A frame that places nothing is naming physical pages already.
    #[test]
    fn a_frame_that_places_no_pages_leaves_them_alone() {
        let frame = frame(Vec::new(), wire());
        let Filled::Ready { plan, .. } =
            fill(&Programs::new(), &frame, &frame.steps[0], 16).expect("a well-formed step")
        else {
            panic!("a host-wire step is never early");
        };
        assert_eq!(plan.kv_page_indices, vec![0, 1, 0, 1]);
    }

    /// A page the frame does not place is refused, not read.
    #[test]
    fn a_page_outside_the_placement_is_refused() {
        let mut sub = wire();
        sub.plan.kv_page_indices = vec![0, 9, 0, 1];
        let frame = frame(vec![3, 2], sub);
        let error = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect_err("page 9 of a two-page placement");
        assert!(
            format!("{error}").contains("working-set page 9"),
            "the refusal names the page: {error}"
        );
    }

    /// A program whose token, position and length come from channels.
    fn envelope_program() -> driver_api::ProgramRegistration {
        use driver::tensor_ir::registry::Port;

        let channel = |id: u32| LaunchChannel {
            id,
            capacity: 1,
            dtype: driver_api::PIE_CHANNEL_DTYPE_U32,
            flags: driver_api::local::PIE_CHANNEL_SEEDED,
            extern_dir: -1,
            readiness: driver_api::local::PIE_READINESS_NEEDS_FULL,
            shape: vec![1],
            extern_name: vec![],
        };
        let port = |port: Port, channel: u32| LaunchPort {
            port: port as u8,
            is_const: false,
            const_dtype: 0,
            channel,
            const_shape: Vec::new(),
            const_data: Vec::new(),
        };
        driver_api::ProgramRegistration {
            program_hash: 0xE0E0,
            launch: LaunchPackage {
                values: Vec::new(),
                channels: vec![channel(0), channel(1), channel(2)],
                ports: vec![
                    port(Port::EmbedTokens, 0),
                    port(Port::Positions, 1),
                    port(Port::KvLen, 2),
                ],
                names: vec![],
                // One empty prologue: the registry refuses a package with no
                // stage at all, and this program's whole behaviour is its
                // ports.
                stages: vec![driver_api::plan::LaunchStage {
                    kind: driver::tensor_ir::registry::Stage::Prologue as u8,
                    ops: vec![],
                    puts: vec![],
                    takes: vec![],
                    reads: vec![],
                }],
                plans: vec![driver_api::plan::LaunchStagePlan::default()],
            },
            ..Default::default()
        }
    }

    /// The DEVICE-GEOMETRY program: the full port set, including the write
    /// descriptor this driver serves.
    ///
    /// Separate from [`envelope_program`] and not an extension of it, because
    /// the two classes are different claims: an envelope binds three ports and
    /// lets the driver page from positions, and this one traces its whole
    /// geometry -- pages, CSR and the write target.
    fn devgeo_program(w_off: bool) -> driver_api::ProgramRegistration {
        use driver::tensor_ir::registry::Port;

        // The CSR is the one port that is not one value per row: a single
        // request needs TWO boundaries.
        let channel = |id: u32| LaunchChannel {
            id,
            capacity: 1,
            dtype: driver_api::PIE_CHANNEL_DTYPE_U32,
            flags: driver_api::local::PIE_CHANNEL_SEEDED,
            extern_dir: -1,
            readiness: driver_api::local::PIE_READINESS_NEEDS_FULL,
            shape: vec![if id == 4 { 2 } else { 1 }],
            extern_name: vec![],
        };
        let port = |port: Port, channel: u32| LaunchPort {
            port: port as u8,
            is_const: false,
            const_dtype: 0,
            channel,
            const_shape: Vec::new(),
            const_data: Vec::new(),
        };
        driver_api::ProgramRegistration {
            program_hash: 0xD0D0,
            launch: LaunchPackage {
                values: Vec::new(),
                channels: (0..if w_off { 7 } else { 6 }).map(channel).collect(),
                ports: {
                    let mut ports = vec![
                        port(Port::EmbedTokens, 0),
                        port(Port::Positions, 1),
                        port(Port::KvLen, 2),
                        port(Port::Pages, 3),
                        port(Port::PageIndptr, 4),
                        port(Port::WSlot, 5),
                    ];
                    if w_off {
                        ports.push(port(Port::WOff, 6));
                    }
                    ports
                },
                names: vec![],
                stages: vec![driver_api::plan::LaunchStage {
                    kind: driver::tensor_ir::registry::Stage::Prologue as u8,
                    ops: vec![],
                    puts: vec![],
                    takes: vec![],
                    reads: vec![],
                }],
                plans: vec![driver_api::plan::LaunchStagePlan::default()],
            },
            ..Default::default()
        }
    }

    /// A bound device-geometry instance, with every channel seeded.
    fn devgeo_bound(seeds: &[(u64, &[u32])]) -> Programs {
        devgeo_bound_with(seeds, true)
    }

    /// The same, with `W_OFF` left unbound when asked: a program that states
    /// WHERE a row's KV goes and not HOW FAR IN.
    fn devgeo_bound_with(seeds: &[(u64, &[u32])], w_off: bool) -> Programs {
        let mut programs = Programs::new();
        let id = programs
            .register_program(&devgeo_program(w_off))
            .expect("a well-formed package");
        for channel in 0..if w_off { 7u64 } else { 6 } {
            let mut ring = u32_ring(channel);
            if channel == 4 {
                ring.shape = vec![2];
                ring.capacity = 1;
            }
            programs.register_channel(&ring).expect("a u32 ring");
        }
        let seeds: Vec<(u64, Vec<u8>)> = seeds
            .iter()
            .map(|&(channel, values)| {
                (
                    channel,
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )
            })
            .collect();
        programs
            .bind_instance(
                id,
                Some(7),
                driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY,
                if w_off {
                    &[0, 1, 2, 3, 4, 5, 6][..]
                } else {
                    &[0, 1, 2, 3, 4, 5][..]
                },
                &seeds,
            )
            .expect("instance 7");
        programs
    }

    /// One device-geometry step: one member, all of its geometry in channels.
    fn devgeo_step() -> StepSubmission {
        StepSubmission {
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY],
            ..envelope_step()
        }
    }

    fn u32_ring(id: u64) -> driver_api::ChannelRegistrationPlan {
        driver_api::ChannelRegistrationPlan {
            channel_id: id,
            dtype: driver_api::PIE_CHANNEL_DTYPE_U32,
            shape: vec![1],
            capacity: 1,
            host_role: driver_api::PIE_CHANNEL_HOST_ROLE_NONE,
            seeded: true,
            extern_dir: driver_api::PIE_CHANNEL_EXTERN_NONE,
            extern_name: Vec::new(),
            driver_id: 0,
            reader_wait_id: 0,
            writer_wait_id: 0,
        }
    }

    /// One decode-envelope step: one member, no geometry on the wire.
    fn envelope_step() -> StepSubmission {
        StepSubmission {
            plan: LaunchPlan {
                token_ids: vec![0],
                position_ids: vec![0],
                qo_indptr: vec![0, 1],
                single_token_mode: true,
                device_resolved_geometry: true,
                ..LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: Vec::new(),
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }
    }

    fn bound(seeds: &[(u64, u32)]) -> Programs {
        let mut programs = Programs::new();
        let id = programs
            .register_program(&envelope_program())
            .expect("a well-formed package");
        for channel in 0..3u64 {
            programs
                .register_channel(&u32_ring(channel))
                .expect("a u32 ring");
        }
        let seeds: Vec<(u64, Vec<u8>)> = seeds
            .iter()
            .map(|&(channel, value)| (channel, value.to_le_bytes().to_vec()))
            .collect();
        programs
            .bind_instance(
                id,
                Some(7),
                driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                &[0, 1, 2],
                &seeds,
            )
            .expect("instance 7");
        programs
    }

    /// A decode states none of its geometry on the wire, and all of it on its
    /// channels: the token it embeds, where that token sits, and -- through
    /// the position -- how much history it attends.
    #[test]
    fn a_decode_envelope_is_filled_from_the_channels_the_last_fire_wrote() {
        let programs = bound(&[(0, 4242), (1, 20), (2, 21)]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        let Filled::Ready { plan, .. } =
            fill(&programs, &frame, &frame.steps[0], 16).expect("a resolvable step")
        else {
            panic!("every channel is seeded, so nothing is early");
        };
        assert_eq!(plan.token_ids, vec![4242], "the token came from channel 0");
        assert_eq!(plan.position_ids, vec![20], "the position from channel 1");
        assert_eq!(plan.qo_indptr, vec![0, 1]);
        assert_eq!(
            plan.kv_page_indices,
            vec![3, 2],
            "position 20 is in working-set page 1, so the fire attends pages 0 and 1"
        );
        assert_eq!(plan.kv_page_indptr, vec![0, 2]);
        assert_eq!(
            plan.kv_last_page_lens,
            vec![5],
            "21 tokens over 16-token pages leaves five in the last one"
        );
    }

    /// A device-geometry fire writes where its PROGRAM said, and the driver
    /// translates the page like every other page of that class.
    #[test]
    fn a_stated_write_target_is_read_and_translated() {
        // token 4242, position 20, kv_len 21, one page (logical 0), CSR [0,1],
        // and a write descriptor naming logical page 1 offset 3.
        let programs = devgeo_bound(&[
            (0, &[4242]),
            (1, &[20]),
            (2, &[21]),
            (3, &[0]),
            (4, &[0, 1]),
            (5, &[1]),
            (6, &[3]),
        ]);
        let frame = frame(vec![3, 2, 1, 0], devgeo_step());
        let Filled::Ready { writes, .. } =
            fill(&programs, &frame, &frame.steps[0], 16).expect("a resolvable step")
        else {
            panic!("every channel is seeded, so nothing is early");
        };
        assert_eq!(
            writes,
            vec![Some((2, 3))],
            "logical page 1 is physical page 2 in this frame's placement, and the \
             offset is the program's"
        );
    }

    /// Half a write descriptor is refused, because the missing half would
    /// have to be invented.
    ///
    /// `W_SLOT` is what turns the descriptor on -- `has_write_desc` is set by
    /// that port alone -- and `W_OFF` is read separately. A program that binds
    /// the first and not the second therefore arrives with slots and no
    /// offsets, and the only ways to serve it are to guess offset 0 (which
    /// overwrites the page's first cell) or to fall back on the position
    /// arithmetic this class exists to replace. Both answer a question the
    /// program did not ask, so it is refused by name instead.
    #[test]
    fn a_write_descriptor_missing_its_offsets_is_refused() {
        let programs = devgeo_bound_with(
            &[
                (0, &[4242]),
                (1, &[20]),
                (2, &[21]),
                (3, &[0]),
                (4, &[0, 1]),
                (5, &[1]),
            ],
            false,
        );
        let frame = frame(vec![3, 2, 1, 0], devgeo_step());
        let Err(Unlaunched::Malformed(why)) = fill(&programs, &frame, &frame.steps[0], 16) else {
            panic!("a slot with no offset is not a write target");
        };
        assert!(
            why.contains("1 slot(s) and 0 offset(s) for 1 row(s)"),
            "the refusal counts both halves, so the guest can see which is short: {why}"
        );
    }

    /// An offset past the page is refused: it would write into the next
    /// page's slots, which belong to somebody else.
    #[test]
    fn a_stated_offset_past_the_page_is_refused() {
        let programs = devgeo_bound(&[
            (0, &[4242]),
            (1, &[20]),
            (2, &[21]),
            (3, &[0]),
            (4, &[0, 1]),
            (5, &[0]),
            (6, &[16]),
        ]);
        let frame = frame(vec![3, 2, 1, 0], devgeo_step());
        let error = fill(&programs, &frame, &frame.steps[0], 16)
            .expect_err("offset 16 of a 16-token page is not a slot");
        assert!(
            format!("{error}").contains("writes offset 16 of a 16-token page"),
            "{error}"
        );
    }

    /// An unfilled descriptor channel is early, not broken.
    #[test]
    fn a_decode_whose_producer_has_not_run_is_early() {
        let programs = bound(&[(1, 20), (2, 21)]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        let filled = fill(&programs, &frame, &frame.steps[0], 16).expect("not an error");
        assert!(
            matches!(filled, Filled::Early { channel: 0 }),
            "the empty token channel is named: {filled:?}"
        );
    }

    /// A wire plan whose caller named several readout rows is refused HERE,
    /// because two lines later there is nothing left to refuse.
    ///
    /// This path drops the sampling table -- deliberately, since the resolved
    /// plan's rows are not the wire's -- so `frames::unserved_in`, where every
    /// other refusal in this driver lives, never sees it. Before this check a
    /// speculative verifier got through admission and faulted inside
    /// `crates/driver`'s interpreter with `logits intrinsic row range exceeds
    /// the forward's readout rows`, which names neither the plan nor the
    /// driver that could not serve it.
    ///
    /// The control is the ordinary table: one row per request is exactly what
    /// A wire's read-out rows are REMAPPED into the layout `fill` builds, not
    /// dropped.
    ///
    /// They used to be dropped, because `fill` rebuilds the row layout and a
    /// wire row index means nothing in the new one. That was true and the
    /// conclusion was wrong: the OFFSET is knowable, because both layouts are
    /// in hand right here. Dropping them lost the one thing a speculative
    /// verifier needs -- which rows of its own it asked to read -- and the
    /// refusal that stood in for it (`a request naming several readout rows`)
    /// is gone with it.
    ///
    /// A row past the request's own count is still refused: read as one it
    /// does have, that is a real distribution belonging to another position.
    #[test]
    fn a_wires_readout_rows_are_remapped_into_the_layout_fill_builds() {
        // `wire()` is TWO requests: rows 0..2 and 2..3.
        let one = {
            let mut sub = wire();
            sub.plan.sampling_indices = vec![1, 0];
            sub.plan.sampling_indptr = vec![0, 1, 2];
            sub
        };
        let frame_one = frame(vec![3, 2, 1, 0], one.clone());
        let Ok(Filled::Ready { plan: filled, .. }) =
            fill(&Programs::default(), &frame_one, &one, 16)
        else {
            panic!("one readout row per request is what this driver gives, and must fill");
        };
        // Request 0's second row and request 1's only one, each in its own
        // request's numbering -- which is what `fill` keeps.
        assert_eq!(filled.sampling_indices, vec![1, 0]);
        assert_eq!(filled.sampling_indptr, vec![0, 1, 2]);

        // BOTH of request 0's rows -- the speculative verifier's shape, which
        // used to be the refusal this test was named for.
        let many = {
            let mut sub = wire();
            sub.plan.sampling_indices = vec![0, 1, 0];
            sub.plan.sampling_indptr = vec![0, 2, 3];
            sub
        };
        let frame_many = frame(vec![3, 2, 1, 0], many.clone());
        let Ok(Filled::Ready { plan: filled, .. }) =
            fill(&Programs::default(), &frame_many, &many, 16)
        else {
            panic!("two readout rows for one request is served now");
        };
        assert_eq!(filled.sampling_indices, vec![0, 1, 0]);
        assert_eq!(filled.sampling_indptr, vec![0, 2, 3]);

        // A row past the request's own count. Request 1 has ONE row, so `1`
        // is already past it -- and read as step-absolute this same table
        // would have been a perfectly ordinary one.
        let stolen = {
            let mut sub = wire();
            sub.plan.sampling_indices = vec![0, 1];
            sub.plan.sampling_indptr = vec![0, 1, 2];
            sub
        };
        let frame_stolen = frame(vec![3, 2, 1, 0], stolen.clone());
        let error = fill(&Programs::default(), &frame_stolen, &stolen, 16)
            .expect_err("a row the request does not have is not translated into one it does");
        assert!(
            format!("{error}").contains("reads out row 1, and has 1 row(s)"),
            "{error}"
        );
    }

    /// A translation table that cannot speak for a row is refused, not read as
    /// the identity.
    ///
    /// # The two empties
    ///
    /// `physical` treats an empty segment as the IDENTITY -- no placement was
    /// made, so a logical page number is its own physical one -- and that is
    /// right for a frame that placed nothing. It is not right for a frame that
    /// HAS a table which does not describe this row: answering `&[]` there
    /// turns a malformed CSR into an identity translation, and an identity
    /// translation on a paged pool is this crate's own worst failure. The fire
    /// reads whatever page carries that number, which is some other
    /// conversation's, and answers fluently.
    ///
    /// The control is the first assertion: the same fill with NO table at all
    /// still succeeds, so this test is about the table and not about the path.
    #[test]
    fn a_translation_that_does_not_describe_a_row_is_refused() {
        let sub = wire();

        // No table: the identity, and served.
        let mut none = frame(vec![3, 2, 1, 0], sub.clone());
        none.kv_translation = Vec::new();
        none.kv_translation_indptr = Vec::new();
        assert!(
            matches!(
                fill(&Programs::default(), &none, &sub, 16),
                Ok(Filled::Ready { .. })
            ),
            "a frame that placed nothing is served by reading its pages straight"
        );

        // A table whose boundaries cross.
        let mut crossed = frame(vec![3, 2, 1, 0], sub.clone());
        crossed.kv_translation = vec![3, 2, 1, 0];
        crossed.kv_translation_indptr = vec![4, 0];
        let said = fill(&Programs::default(), &crossed, &sub, 16)
            .expect_err("a CSR that runs backwards is not read as the identity")
            .to_string();
        assert!(
            said.contains("translation"),
            "the refusal names the table: {said}"
        );

        // A table whose end runs past what it indexes.
        let mut past = frame(vec![3, 2, 1, 0], sub.clone());
        past.kv_translation = vec![3, 2];
        past.kv_translation_indptr = vec![0, 9];
        assert!(
            fill(&Programs::default(), &past, &sub, 16).is_err(),
            "a CSR ending past its table is not read as the identity"
        );

        // A table too short to name this row at all.
        let mut short = frame(vec![3, 2, 1, 0], sub.clone());
        short.kv_translation = vec![3, 2, 1, 0];
        short.kv_translation_indptr = vec![0];
        assert!(
            fill(&Programs::default(), &short, &sub, 16).is_err(),
            "a CSR with no boundary for this row is not read as the identity"
        );
    }

    /// The `kv_len` port is honoured by being checked against the positions.
    ///
    /// # Why a check and not a read
    ///
    /// This seam claims `PIE_DECODE_ENVELOPE_PORTS`, which names three ports:
    /// `EMBED_TOKENS`, `POSITIONS` and `KV_LEN`. The first two are read
    /// straight off the resolution. The third is DERIVED from the position,
    /// for the ordering reason `fill` gives — a row writes where its position
    /// says, and reading one page fewer than it writes is an attention over a
    /// page this fire itself filled.
    ///
    /// A derivation is not a reading, and the difference is invisible while
    /// the two agree. They agree for every decode the engine builds: a guest's
    /// `kv_len.put(&next_length)` and its `positions.put(&length)` come off
    /// the same counter, which is what the first assertion here measures. A
    /// guest that parted them — a sliding window, an evicted cache — used to
    /// get an attention over a span it had not asked for, silently. It gets a
    /// refusal naming both numbers now.
    #[test]
    fn a_kv_len_that_disagrees_with_the_positions_is_refused() {
        // Position 20 with a page size of 16 puts the last page at length 5,
        // and 21 tokens of history is exactly that. Agreeing, and served.
        let agreeing = bound(&[(0, 4242), (1, 20), (2, 21)]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        assert!(
            matches!(
                fill(&agreeing, &frame, &frame.steps[0], 16),
                Ok(Filled::Ready { .. })
            ),
            "the ordinary decode agrees with itself and must still fill"
        );

        // The same fire with a history the guest states SHORTER than its own
        // position: a window. Refused, by both numbers.
        let parted = bound(&[(0, 4242), (1, 20), (2, 8)]);
        let said = fill(&parted, &frame, &frame.steps[0], 16)
            .expect_err("a stated history that is not the derived one is not served")
            .to_string();
        for part in ["8", "5"] {
            assert!(
                said.contains(part),
                "the refusal names both numbers, and omits `{part}`: {said}"
            );
        }
    }
}
