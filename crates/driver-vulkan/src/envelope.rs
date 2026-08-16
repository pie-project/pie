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
    Ready {
        /// The rebuilt plan.
        plan: Box<LaunchPlan>,
        /// One byte per request of that plan: non-zero where the program
        /// traced its own pages. Carried beside the plan rather than on it
        /// because `LaunchPlan` is the WIRE's vocabulary and this is a fact
        /// about how the plan was built, not about what was sent.
        traced: Vec<u8>,
        /// Where each request's rows write, when the program stated it; see
        /// [`crate::resources::Request::writes`]. One entry per request, empty
        /// where the placement is the derivation.
        writes: Vec<Vec<(u32, u32)>>,
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
/// is a class this driver serves -- so the guess stopped nothing and instead
/// sent the member down the path that reads geometry out of the WIRE PLAN. A
/// device-resolved member leaves those tables empty, so the guess surfaced
/// later as `request 0 has no page span in a CSR of 0 entries`: true, useless,
/// and blaming the plan for a decision made here.
///
/// An empty `sub_batch_indptr` stays `Host`: `StepSubmission::validate` admits
/// an absent table, and a step with no sub-batching has no other class it
/// could be. What is refused is a table that EXISTS and leaves this member
/// out, or one whose class list is shorter than its own CSR.
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
/// The rows request `r` names in the wire plan's sampling table.
///
/// Empty for a plan with no table, which means every request reads out its own
/// last row -- the decode case.
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

/// The rows request `r` of a RESOLVED geometry reads, in its own numbering.
///
/// Same CSR shape and same convention as [`sampling_rows`]; separate because
/// a `driver::Geometry` is not a `LaunchPlan`, and an empty slice means the
/// instance named none rather than that it named zero.
fn sampling_rows_of(geometry: &driver::FireGeometry, r: usize) -> &[u32] {
    let (Some(&lo), Some(&hi)) = (
        geometry.sampling_indptr.get(r),
        geometry.sampling_indptr.get(r + 1),
    ) else {
        return &[];
    };
    if hi < lo {
        return &[];
    }
    geometry
        .sampling_indices
        .get(lo as usize..hi as usize)
        .unwrap_or(&[])
}

/// One row of allow-bytes as the alternating runs the wire spells.
///
/// The encoding starts with a FALSE run and alternates, which is the reading
/// [`crate::frames::mask_rows_of`] uses in reverse -- an even index is a
/// forbidden run, an odd one allowed. A row that begins allowed therefore
/// opens with a zero-length false run, and that is not a special case.
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

fn translation(frame: &FrameSubmission, row: u32) -> &[u32] {
    let row = row as usize;
    match (
        frame.kv_translation_indptr.get(row),
        frame.kv_translation_indptr.get(row + 1),
    ) {
        (Some(&lo), Some(&hi)) if hi >= lo && hi as usize <= frame.kv_translation.len() => {
            &frame.kv_translation[lo as usize..hi as usize]
        }
        _ => &[],
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
    // The read-out rows, REMAPPED. `fill` rebuilds the row layout, so a wire
    // row index means nothing here -- but the offset does: request `r`'s rows
    // start at `base` in the new layout and at `lo` in the old.
    let (mut samples, mut sample_indptr) = (Vec::new(), vec![0u32]);
    // The DENSE mask a device-geometry program resolves, re-encoded as the
    // runs the rest of this driver reads. Two vectors rather than one because
    // `mask_indptr` is per REQUEST and the runs are per ROW.
    let (mut masks, mut mask_indptr) = (Vec::new(), vec![0u32]);
    let mut any_mask = false;
    // Which requests of the rebuilt plan have pages the PROGRAM traced. One
    // byte per request, in the plan's own order, because a step may mix a
    // device-geometry member with a host one and the two are placed
    // differently -- see `resources::Request::traced`.
    let mut traced: Vec<u8> = Vec::new();
    // Where each request's rows write, when the program STATED it. One entry
    // per request in the plan's order; an empty inner vector is "derive it",
    // which is every request the scheduler paged.
    let mut writes: Vec<Vec<(u32, u32)>> = Vec::new();

    for (member, &row) in sub.roster_rows.iter().enumerate() {
        let segment = translation(frame, row);
        let class = class_of(sub, member).ok_or_else(|| {
            Unlaunched::Malformed(format!(
                "step member {member} is in no sub-batch: {} boundaries covering members \
                 0..{}, and {} class(es)",
                sub.sub_batch_indptr.len(),
                sub.sub_batch_indptr.last().copied().unwrap_or(0),
                sub.sub_batch_class.len()
            ))
        })?;
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
            for (r, request) in resolved.qo_indptr.windows(2).enumerate() {
                let (lo, hi) = (request[0] as usize, request[1] as usize);
                if lo > hi || hi > resolved.token_ids.len() || hi > resolved.position_ids.len() {
                    return Err(Unlaunched::Malformed(format!(
                        "instance {id} resolves a request over rows {lo}..{hi} of {} token(s)",
                        resolved.token_ids.len()
                    )));
                }
                tokens.extend_from_slice(&resolved.token_ids[lo..hi]);
                // The rows this instance RESOLVED it would read, in the same
                // per-request numbering the host-wire branch below carries.
                //
                // This branch used to carry none and lean on `Request::read`,
                // which reads the last row when the table is empty. That is
                // the right answer for a one-row decode and only for a
                // one-row decode -- `driver::resolve` states `span - 1` per
                // request, so a request of several rows resolves several, and
                // dropping them reads one. It is the same defect as dropping
                // the wire's table (see the note at the end of `fill`), which
                // faulted as "logits intrinsic row range exceeds the
                // forward's readout rows"; the two branches now agree.
                for &row in sampling_rows_of(&resolved, r) {
                    let row = row as usize;
                    if row >= hi - lo {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} resolves a read-out of its own row {row}, past \
                             the {} row(s) request {r} spans",
                            hi - lo
                        )));
                    }
                    samples.push(u32::try_from(row).unwrap_or(u32::MAX));
                }
                // The CSR still needs a boundary per request: one short would
                // give the next request these rows.
                sample_indptr.push(u32::try_from(samples.len()).unwrap_or(u32::MAX));
                positions.extend_from_slice(&resolved.position_ids[lo..hi]);
                qo.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
                // A device-geometry program STATES its pages, so they are read
                // off the resolution and translated. `has_kv_family` is what
                // says the family was bound at all; a program of this class
                // that bound none is refused rather than quietly given the
                // envelope's arithmetic, because the two answer differently
                // the moment a guest's pages are not its own history in order
                // -- which is the whole reason to trace them.
                let stated = class == driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY;
                if stated && !resolved.has_kv_family {
                    return Err(Unlaunched::Malformed(format!(
                        "instance {id} is device-geometry and binds no KV family: its \
                         pages would have to be guessed from its positions, which is a \
                         different fire from the one it traced"
                    )));
                }
                // The pages a decode attends are its own history, and the
                // last position it writes says how much of that there is.
                // Taken from the POSITIONS rather than from `kv_len`,
                // because a row writes where its position says and reading
                // one page fewer than it writes is an attention over a page
                // this fire itself filled.
                let last = resolved.position_ids[lo..hi].iter().copied().max();
                if stated {
                    let (plo, phi) = match (
                        resolved.kv_page_indptr.get(r),
                        resolved.kv_page_indptr.get(r + 1),
                    ) {
                        (Some(&plo), Some(&phi)) => (plo as usize, phi as usize),
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
                            "instance {id} resolves request {r} over pages {plo}..{phi} of {}",
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
                    // The write target, READ rather than derived.
                    //
                    // `Frame::of` derives where each row writes -- page
                    // `position / page_size` of the request's own list, offset
                    // `position % page_size` -- because for every fire the
                    // scheduler pages, that arithmetic IS the placement. A
                    // device-geometry program states it instead, through
                    // `Port::WSlot` and `Port::WOff`, in the same LOGICAL page
                    // numbers it states its span in.
                    //
                    // The two disagree, and beam search is where. This driver
                    // first CHECKED the statement against the derivation, and
                    // the check faulted on the second fire of every beam:
                    //
                    //     instance 1 request 1 writes position 1 to page 7
                    //     offset 2, and its own span places that position at
                    //     page Some(7) offset 1
                    //
                    // Two lanes of one instance share the page they forked
                    // from and take separate SLOTS inside it. The derivation
                    // reads the offset off the POSITION, and both lanes are at
                    // the same position, so it can only ever name one slot for
                    // both -- which is the same fact that made `Frame::of`
                    // refuse these two lanes as sharing a page. The statement
                    // is the answer; the derivation was the wrong question.
                    let mut mine = Vec::new();
                    if resolved.has_write_desc {
                        for idx in lo..hi {
                            let pos = resolved.position_ids[idx];
                            let (Some(&stated_page), Some(&stated_off)) =
                                (resolved.w_page.get(idx), resolved.w_off.get(idx))
                            else {
                                return Err(Unlaunched::Malformed(format!(
                                    "instance {id} binds a write descriptor of {} page(s) \
                                     and {} offset(s) for {} row(s)",
                                    resolved.w_page.len(),
                                    resolved.w_off.len(),
                                    resolved.position_ids.len()
                                )));
                            };
                            if stated_off >= page {
                                return Err(Unlaunched::Malformed(format!(
                                    "instance {id} request {r} writes position {pos} to \
                                     offset {stated_off} of a page of {page} slot(s)"
                                )));
                            }
                            mine.push((physical(segment, stated_page)?, stated_off));
                        }
                    }
                    writes.push(mine);
                } else {
                    let live = last.map_or(0, |p| pages_for(p.saturating_add(1), page));
                    for logical in 0..live {
                        pages.push(physical(segment, logical)?);
                    }
                }
                page_indptr.push(u32::try_from(pages.len()).unwrap_or(u32::MAX));
                if stated {
                    // A device-geometry program states its own span, and its
                    // pages are not required to be its history in order -- an
                    // evicted cache or a sliding window is exactly what this
                    // class exists to express -- so the derivation below is
                    // not a second opinion here, it is a different question.
                    // The STATED value is the answer, and its absence is a
                    // refusal rather than a fallback to the derivation.
                    let Some(&live) = resolved.kv_last_page_lens.get(r) else {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} is device-geometry and states no last-page \
                             length for request {r}"
                        )));
                    };
                    if live > page {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} states a last page of {live} rows in a pool \
                             whose pages hold {page}"
                        )));
                    }
                    lens.push(live);
                    // The dense mask this program resolved, one byte per lane,
                    // cut into the rows of THIS request and re-encoded as
                    // runs. `mask_key_len` is the row pitch the resolver
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
                } else {
                    lens.push(last.map_or(0, |p| driver::last_page_len(p.saturating_add(1), page)));
                }
                // A boundary per request either way: a table one short would
                // give the next request this one's runs.
                mask_indptr.push(u32::try_from(masks.len()).unwrap_or(u32::MAX));
                traced.push(u8::from(stated));
                if !stated {
                    writes.push(Vec::new());
                }
            }
            continue;
        }

        let (first, last) = member_requests(&sub.program_row_indptr, member, wire_rows)
            .ok_or_else(|| {
                Unlaunched::Malformed(format!(
                    "member {member}'s rows are not placed by `program_row_indptr` {:?} \
                     among {wire_rows} request(s)",
                    sub.program_row_indptr
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
            // A read-out row is stated in the REQUEST's own numbering, and is
            // carried in it. `driver::resolve` writes `span - 1` per request,
            // `scheduler::wire` asserts `index < row_len` while merging, and
            // this leaves it alone -- so one numbering runs from the engine's
            // geometry all the way to `resources::Request::samples`, and
            // there is no seam where it changes.
            //
            // It used to be rebased through `lo`, which is the identity for a
            // single request and a NEIGHBOUR's distribution for any other.
            //
            // A row past the request's own width would read a real
            // distribution belonging to another conversation.
            for &row in sampling_rows(&sub.plan, r) {
                let row = row as usize;
                if row >= hi - lo {
                    return Err(Unlaunched::Malformed(format!(
                        "request {r} reads out its own row {row}, past the {} row(s) it \
                         spans ({lo}..{hi})",
                        hi - lo
                    )));
                }
                samples.push(u32::try_from(row).unwrap_or(u32::MAX));
            }
            sample_indptr.push(u32::try_from(samples.len()).unwrap_or(u32::MAX));
            // The host branch names no resolved runs, and still needs its
            // boundary: without one a fire that mixed a host member with a
            // device-geometry member would hand the host member's requests the
            // device member's mask rows.
            mask_indptr.push(u32::try_from(masks.len()).unwrap_or(u32::MAX));
            positions.extend_from_slice(&sub.plan.position_ids[lo..hi]);
            qo.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
            let (plo, phi) = match (
                sub.plan.kv_page_indptr.get(r),
                sub.plan.kv_page_indptr.get(r + 1),
            ) {
                (Some(&plo), Some(&phi)) if phi >= plo => (plo as usize, phi as usize),
                _ => {
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
            traced.push(0);
            writes.push(Vec::new());
        }
    }

    out.token_ids = tokens;
    out.position_ids = positions;
    if any_mask {
        if !out.masks.is_empty() {
            // The wire's rows and the resolution's rows are two masks for one
            // fire, and there is no reading of that which is not a guess:
            // overwriting drops the wire's, appending numbers them against a
            // CSR that describes neither. Refused by name.
            return Err(Unlaunched::Malformed(format!(
                "this step carries {} mask row(s) on the wire and a device-geometry \
                 member resolved {} more",
                out.masks.len(),
                masks.len()
            )));
        }
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
    // rows of its own it asked to read. Leaving it dropped is what made the
    // gather fall back to one row per request and the program fault with
    // "logits intrinsic row range exceeds the forward's readout rows".
    out.sampling_indices = samples;
    out.sampling_indptr = sample_indptr;
    Ok(Filled::Ready {
        plan: Box::new(out),
        traced,
        writes,
    })
}

#[cfg(test)]
mod tests {
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

    /// A step that says something about its members and leaves one out is
    /// refused, because the alternative is serving it as a class nobody chose.
    #[test]
    fn a_member_in_no_sub_batch_is_refused_rather_than_called_host() {
        let mut sub = wire();
        sub.sub_batch_indptr = vec![0, 0];
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let error = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect_err("a member in no sub-batch has no class to be served as");
        assert!(
            format!("{error}").contains("step member 0 is in no sub-batch"),
            "{error}"
        );
    }

    /// The same hole from the other side: the window exists and the class list
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

    /// An ABSENT table still reads as `Host`: the wire contract admits it, and
    /// a step with no sub-batching has no other class it could be.
    #[test]
    fn a_step_with_no_sub_batch_table_at_all_is_host() {
        let mut sub = wire();
        sub.sub_batch_indptr = Vec::new();
        sub.sub_batch_class = Vec::new();
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let Filled::Ready { plan, .. } = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect("an absent sub-batch table is a host step")
        else {
            panic!("a host-wire step is never early: it names all of its own geometry");
        };
        assert_eq!(
            plan.kv_page_indices,
            vec![3, 2, 3, 2],
            "the host path ran and translated the working set"
        );
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

    /// A read-out row is stated in ITS OWN request's numbering.
    ///
    /// # The bug this pins, and why nothing else could see it
    ///
    /// `fill` read `sampling_indices` as rows of the whole fire and rebased
    /// them through the request's start (`base + (row - lo)`). The engine
    /// states them per request: `scheduler::wire` appends each request's
    /// indices unchanged and asserts `index < row_len` while doing it, and
    /// `driver::resolve` writes `span - 1` with a test that spells out "both
    /// relative to their own request".
    ///
    /// Subtracting `lo` is the identity when `lo == 0`, which is every
    /// single-request plan -- so every test in this crate agreed with the
    /// wrong reading. What disagreed was eight conversations at once
    /// (`vulkan_many_conversations`), where the second request's own last row
    /// is 91 and so is the first's, and the fire was refused: "request 1
    /// reads out row 91, which is not in its own rows 92..184". The plan was
    /// right.
    ///
    /// The envelope is where the numbering CHANGES: `frames::sample_rows_of`
    /// reads what this function writes, so it correctly rebases through `lo`
    /// -- and that is why the fix belongs here and only here.
    #[test]
    fn a_read_out_row_is_numbered_within_its_own_request() {
        let mut sub = wire();
        // Two requests, rows 0..2 and 2..3, each reading its own last row.
        sub.plan.sampling_indices = vec![1, 0];
        sub.plan.sampling_indptr = vec![0, 1, 2];
        let frame = frame(vec![3, 2, 1, 0], sub);
        let Filled::Ready { plan, .. } =
            fill(&Programs::new(), &frame, &frame.steps[0], 16).expect("a well-formed step")
        else {
            panic!("a host-wire step is never early");
        };
        assert_eq!(
            plan.sampling_indices,
            vec![1, 0],
            "each request's own last row, carried unchanged; rebasing through \
             `lo` refuses request 1 outright for naming row 0, which under \
             that reading belongs to request 0"
        );
        assert_eq!(plan.sampling_indptr, vec![0, 1, 2]);
    }

    /// A read-out past the request's own width is refused, not rebased.
    #[test]
    fn a_read_out_row_past_its_request_is_refused() {
        let mut sub = wire();
        // Request 1 spans one row and names a second.
        sub.plan.sampling_indices = vec![1, 1];
        sub.plan.sampling_indptr = vec![0, 1, 2];
        let frame = frame(vec![3, 2, 1, 0], sub);
        let error = fill(&Programs::new(), &frame, &frame.steps[0], 16)
            .expect_err("a one-row request naming its second row");
        assert!(
            format!("{error}").contains("past the 1 row(s)"),
            "the refusal names the width: {error}"
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

    /// The same program plus a read-out port: a speculative decode states
    /// which of its rows it wants distributions for.
    fn speculative_program() -> driver_api::ProgramRegistration {
        use driver::tensor_ir::registry::Port;

        let mut registration = envelope_program();
        let launch = &mut registration.launch;
        launch.channels.push(LaunchChannel {
            id: 3,
            capacity: 1,
            dtype: driver_api::PIE_CHANNEL_DTYPE_U32,
            flags: driver_api::local::PIE_CHANNEL_SEEDED,
            extern_dir: -1,
            readiness: driver_api::local::PIE_READINESS_NEEDS_FULL,
            shape: vec![2],
            extern_name: vec![],
        });
        launch.ports.push(LaunchPort {
            port: Port::Readout as u8,
            is_const: false,
            const_dtype: 0,
            channel: 3,
            const_shape: Vec::new(),
            const_data: Vec::new(),
        });
        for channel in &mut launch.channels {
            if channel.id < 2 {
                channel.shape = vec![3];
            }
        }
        registration.program_hash = 0xE0E1;
        registration
    }

    /// `bound`, but every channel carries a vector rather than one value.
    fn speculative(seeds: &[(u64, Vec<u32>)]) -> Programs {
        let mut programs = Programs::new();
        let id = programs
            .register_program(&speculative_program())
            .expect("a well-formed package");
        for (channel, values) in seeds {
            let mut plan = u32_ring(*channel);
            plan.shape = vec![u32::try_from(values.len()).unwrap()];
            programs.register_channel(&plan).expect("a u32 ring");
        }
        let seeds: Vec<(u64, Vec<u8>)> = seeds
            .iter()
            .map(|(channel, values)| {
                (
                    *channel,
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )
            })
            .collect();
        programs
            .bind_instance(
                id,
                Some(7),
                driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                &[0, 1, 2, 3],
                &seeds,
            )
            .expect("instance 7");
        programs
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

    /// A decode envelope reading SEVERAL rows gets several, not just the last.
    ///
    /// # The blind spot this closes
    ///
    /// This branch used to carry no read-out table at all and lean on
    /// `Request::read`, which returns the last row when the table is empty.
    /// For the one-row decode above that is the same answer, which is why
    /// every test here agreed with it -- the fixture had one row, so "all the
    /// rows it asked for" and "the last row" were the same list.
    ///
    /// A speculative decode is the case where they differ: three rows on the
    /// wire and a read-out port naming rows 0 and 2. The old branch read row
    /// 2 alone, which is the shape of fault the host-wire branch produced
    /// when its table was dropped -- "logits intrinsic row range exceeds the
    /// forward's readout rows".
    #[test]
    fn a_decode_envelope_reading_several_rows_carries_all_of_them() {
        let programs = speculative(&[
            (0, vec![4242, 4243, 4244]),
            (1, vec![20, 21, 22]),
            (2, vec![23]),
            (3, vec![0, 2]),
        ]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        let Filled::Ready { plan, .. } =
            fill(&programs, &frame, &frame.steps[0], 16).expect("a resolvable step")
        else {
            panic!("every channel is seeded, so nothing is early");
        };
        assert_eq!(plan.token_ids, vec![4242, 4243, 4244]);
        assert_eq!(
            plan.sampling_indices,
            vec![0, 2],
            "the instance resolved a read-out of rows 0 and 2; carrying none \
             leaves `Request::read` to answer with row 2 alone"
        );
        assert_eq!(plan.sampling_indptr, vec![0, 2]);
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
}
