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
//! # Why this exists at all, said in the words the failure used
//!
//! `serve/load.rs` reported `device_geometry_port_mask: 0` for as long as
//! there was nothing here, and the zero was HONEST then -- a driver that
//! claims a port it cannot read is a driver that answers a question it never
//! asked. The engine answered the zero by folding the geometry on the host,
//! which cannot know `EmbedTokens` because the token is a cell no host wrote:
//!
//! ```text
//! decode envelope on a driver without device geometry ports (mask 0x0,
//! needs 0x25): falling back to host-evaluated serialized execution
//! ... EmbedTokens is not host-derivable: channel 0 has no host-known value
//! ```
//!
//! So the remedy was to build the machinery rather than to widen the claim,
//! which is what `driver-wgpu/src/envelope.rs` and `driver-vulkan`'s copy
//! each concluded before this one. This file is the Metal port of the first,
//! whose channel ring is HOST memory exactly as this driver's is.
//!
//! # The pages, and the bug that is invisible with one conversation
//!
//! `LaunchPlan::kv_page_indices` are indices into the request's own working
//! set, and the frame's `kv_translation` says which physical page each one
//! is. A driver that skips the translation reads page 0 of the pool for every
//! conversation. With one conversation in flight that is accidentally right,
//! which is exactly why it survives a single-request test: the first
//! conversation's working set is placed at pool pages 0, 1, 2 and the
//! identity is consistent with itself. With two, the second conversation
//! attends the first one's keys and answers fluently in the first one's
//! context.
//!
//! This driver used to apply no translation at all -- `pools::kv::translate`
//! is a BOUNDS CHECK on the frame's table and never rewrote a page number --
//! so it held the accidental identity. The translation is applied here, for
//! every class, because doing it for the envelope alone would mean two
//! readings of one table living side by side.
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
//!
//! # What is REFUSED, by name
//!
//! `PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY` is the further class: a program that
//! traces its whole geometry in-graph and states its pages, its CSR, its
//! dense attention mask and its per-row write descriptor on channels. The
//! wgpu port serves it; this one does not, and says so rather than serving it
//! as an envelope. The reason is not effort, it is that three of those four
//! statements have NO CONSUMER in this driver: `serve/launch.rs` derives
//! every row's write target from its position (`w_page`/`w_off` are computed
//! there, not read), and nothing in this crate's lowering reads
//! `LaunchPlan::masks` -- a custom mask reaches the Metal text through the
//! region table's `MASK` bit and not through the plan. Filling those fields
//! and firing anyway would be a fire that ignores what the program said,
//! which is the one outcome worse than a refusal.

use driver_api::{FrameSubmission, LaunchPlan, StepSubmission};

use crate::error::{Error, Result};

/// What filling a step's geometry produced.
#[derive(Debug)]
pub enum Filled {
    /// The step can be fired, with this plan in place of the wire's.
    Ready {
        /// The geometry this step fires over.
        plan: Box<LaunchPlan>,
    },
    /// A descriptor channel is empty: the program that fills it has not run.
    ///
    /// Not a failure, and not the same thing as a refusal. The producer is
    /// usually the step BEFORE this one in the same frame, so the remedy is
    /// to fire that one first; nothing of THIS step has happened, which is
    /// what makes stopping here safe.
    ///
    /// v14 admission is supposed to make this unreachable at fire time --
    /// the scheduler does not seal a step whose producer has not run -- so a
    /// caller that sees it fails the step's members rather than re-posting.
    /// `driver-vulkan`'s seam says the same thing at the same point, and
    /// RETRY is not an outcome the terminal cell has.
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
/// A member outside every sub-batch window used to read back as `Host` in the
/// drivers this is ported from, which is a class every driver knows how to
/// serve -- so the guess did not stop anything, it just sent the member down
/// a path that reads geometry out of the WIRE PLAN. A device-geometry member
/// states its pages in a channel and leaves the wire plan's tables empty, so
/// the guess surfaced four steps later as `request 0 has no page span in a
/// CSR of 0 entries`: true, useless, and blaming the plan for a decision made
/// here.
///
/// An empty `sub_batch_indptr` is different and stays `Host`:
/// `StepSubmission::validate` admits an absent table, and a step with no
/// sub-batching has no other class it could be. What is refused is a table
/// that EXISTS and does not cover this member, or one whose class list is
/// shorter than its own CSR -- both of which mean the step said something
/// about its members and left this one out.
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

/// Whether any member of this step has its geometry on a channel.
///
/// The question `serve/launch.rs` asks before it decides how to drive the
/// frame: a frame of ordinary host-wire steps can be encoded whole and waited
/// for afterwards, and a frame with one device-resolved member cannot,
/// because that member's tokens are what the step BEFORE it put on a channel.
/// A step whose class table does not describe some member answers `true`
/// here, deliberately: the safe reading of "I cannot tell" is the one that
/// costs pipelining rather than the one that fires a step over a cell that is
/// not there yet, and [`fill`] refuses that member by name a moment later.
#[must_use]
pub fn resolves_on_device(sub: &StepSubmission) -> bool {
    (0..sub.roster_rows.len())
        .any(|member| class_of(sub, member) != Some(driver_api::PIE_GEOMETRY_CLASS_HOST))
}

/// Which requests of the wire plan belong to batch member `member`.
///
/// `program_row_indptr` is the step's own attribution CSR -- member `p` owns
/// request rows `[indptr[p], indptr[p + 1])`.
///
/// # Why an unusable CSR is `None` rather than "all of them"
///
/// The fallback for an ABSENT table is the single-member case and is right.
/// The same fallback for a table that is present and does not describe this
/// member is not: on `[0, 1, 2, 9]` with three requests, members 0 and 1
/// answer correctly and member 2 takes all three requests' rows -- one member
/// out of three, silently serving another conversation's geometry, in a frame
/// whose other members are fine. `driver-vulkan` found that; `driver-wgpu`
/// carried it for another eleven weeks. So the two claims are kept apart and
/// the second is refused by the caller, by name.
fn member_requests(
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

/// The rows request `r` names in the wire plan's sampling table.
///
/// Empty for a plan with no table, which means every request reads out its
/// own last row -- the decode case. Empty too for a CSR that does not name
/// this request, which `validate_geometry` has already had its say about;
/// answering with another request's span would be the one wrong answer
/// available here.
fn sampling_rows(plan: &LaunchPlan, r: usize) -> &[u32] {
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

/// One roster row's slice of the frame's page translation.
///
/// # Why an empty answer is not one answer
///
/// [`physical`] reads an empty segment as the IDENTITY: no placement was
/// made, so a fire's logical page number is its physical one. That is right
/// for a frame that placed nothing, which is what this crate's own tests
/// build.
///
/// It is not right for a frame that HAS a translation table which does not
/// describe this row -- a CSR whose boundaries cross, or whose end runs past
/// the table it indexes. Answering `&[]` there turns a malformed table into
/// an identity translation, and an identity translation on a paged pool is
/// the failure this module is written against: the fire reads whatever page
/// carries that logical number, which is some other conversation's, and
/// answers fluently.
///
/// # Errors
///
/// [`Error::Unserved`] for a translation CSR that does not describe `row`.
fn translation(frame: &FrameSubmission, row: u32) -> Result<&[u32]> {
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
        (Some(&lo), Some(&hi)) => Err(refused(format!(
            "this frame's translation of row {row} is {lo}..{hi} of a table holding {} page(s)",
            frame.kv_translation.len()
        ))),
        _ => Err(refused(format!(
            "this frame's translation names {} boundaries, which do not describe row {row}",
            frame.kv_translation_indptr.len()
        ))),
    }
}

/// One working-set page as the physical page it is placed in.
///
/// An EMPTY translation means the frame states none, and the only honest
/// reading of a frame that names no placement is that the pages it names are
/// already physical.
///
/// # Errors
///
/// [`Error::Unserved`] naming the page the frame's translation does not
/// place. `u32::MAX` is the ABI's "reserved but not materialized" and is a
/// page nothing may address, so it is refused here rather than handed to the
/// pool as a page index of four billion.
fn physical(segment: &[u32], logical: u32) -> Result<u32> {
    if segment.is_empty() {
        return Ok(logical);
    }
    segment
        .get(logical as usize)
        .copied()
        .filter(|&page| page != u32::MAX)
        .ok_or_else(|| {
            refused(format!(
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

/// The one refusal shape this module raises.
///
/// `Unserved` and not `Program`: every failure here is a statement about the
/// FRAME -- a table that does not close, a page nothing placed, a member with
/// no class -- rather than about a shader or a device, and `Error::Unserved`
/// is what stamps `driver-metal: {what}:` in front of it so an operator can
/// tell which of the four layers refused.
fn refused(message: String) -> Error {
    Error::Unserved {
        what: "fill",
        message,
    }
}

/// Read one instance's device-resolved geometry off its channels.
///
/// A PEEK, not a take: the ports that consume are consumed once, later, by
/// the interpreter's own port loop when the program fires. Reading twice
/// would drop a cell, and the symptom is a fire silently using the
/// fire-after-next's tokens.
///
/// `page` is the KV page size, which the `kv_len` port's contract needs: the
/// port carries a LENGTH and the plan wants how much of the last page that
/// length fills.
///
/// The whole body is `driver::resolve`, as `driver_wgpu::programs::Programs::geometry`
/// is, because the channel plane is SHARED -- `crate::channel` is `pub use
/// driver::*` and nothing else -- so this driver reads its cells with the
/// same reader the other two use rather than a second one that agrees by
/// inspection.
///
/// # Errors
///
/// [`Error::Unserved`] for an instance id the registry does not hold, or one
/// whose program has gone. Both name the id, because a peek that fails by
/// not existing tells an operator nothing about which conversation is stuck.
pub fn geometry(
    registry: &crate::channel::Registry,
    instance: u64,
    page: u32,
) -> Result<driver::Resolution> {
    let inst = registry
        .instance(instance)
        .ok_or_else(|| refused(format!("no instance {instance}")))?;
    let program = registry.program(inst.program_id).ok_or_else(|| {
        refused(format!(
            "instance {instance} names program {} which is gone",
            inst.program_id
        ))
    })?;
    Ok(driver::resolve(&program.plan, &inst.interp, page))
}

/// This step's plan with every member's geometry known and every page
/// physical.
///
/// # Errors
///
/// [`Error::Unserved`] for a CSR that does not close, a page the frame's
/// translation does not place, a member whose class this driver does not
/// serve, or a descriptor channel whose cell is not a geometry index.
pub fn fill(
    registry: &crate::channel::Registry,
    frame: &FrameSubmission,
    sub: &StepSubmission,
    page: u32,
) -> Result<Filled> {
    let wire_rows = sub.plan.qo_indptr.len().saturating_sub(1);
    let mut out = sub.plan.clone();
    let (mut tokens, mut positions, mut qo) = (Vec::new(), Vec::new(), vec![0u32]);
    let (mut pages, mut page_indptr, mut lens) = (Vec::new(), vec![0u32], Vec::new());
    // The read-out rows, kept in the numbering they arrive in: request `r`'s
    // value `k` is that request's own row `k`, which is what
    // `lowering::frame::sampled_rows` reads and translates into fire rows. A
    // table restated in fire rows here would be translated twice.
    let (mut samples, mut sample_indptr) = (Vec::new(), vec![0u32]);

    for (member, &row) in sub.roster_rows.iter().enumerate() {
        let segment = translation(frame, row)?;
        let class = class_of(sub, member).ok_or_else(|| {
            refused(format!(
                "step member {member} is in no sub-batch: {} boundaries covering members \
                 0..{}, and {} class(es)",
                sub.sub_batch_indptr.len(),
                sub.sub_batch_indptr.last().copied().unwrap_or(0),
                sub.sub_batch_class.len()
            ))
        })?;
        // The further class, refused BY NAME and not served as its smaller
        // neighbour. See this module's header: a device-geometry program
        // states its pages, its CSR, its dense mask and its write descriptor,
        // and this driver has a consumer for the first two only. Serving it
        // as an envelope would derive from the positions the three things the
        // program traced in-graph precisely so that they would NOT be derived
        // -- an evicted cache, a sliding window, a forked beam -- and the
        // fire would run, and be wrong, and say nothing.
        if class == driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY {
            return Err(refused(format!(
                "step member {member} is geometry class {} (device geometry), which this \
                 driver does not serve: it would have to honour a stated write descriptor \
                 and a dense attention mask, and `serve::launch` derives every row's write \
                 target from its position and reaches a custom mask through the region \
                 table rather than through the plan. `serve::load` claims \
                 PIE_DECODE_ENVELOPE_PORTS and no more, so a program of this class should \
                 not have reached here at all",
                driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY
            )));
        }
        if class == driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE {
            let id = *frame.instance_ids.get(row as usize).ok_or_else(|| {
                refused(format!(
                    "step member {member} names roster row {row} of {}",
                    frame.instance_ids.len()
                ))
            })?;
            let resolved = match geometry(registry, id, page)? {
                driver::Resolution::Ready(geometry) => geometry,
                driver::Resolution::NotReady { channel } => {
                    return Ok(Filled::Early { channel });
                }
                driver::Resolution::Failed { message } => {
                    return Err(refused(format!(
                        "instance {id} resolves no geometry: {message}"
                    )));
                }
            };
            for (which, request) in resolved.qo_indptr.windows(2).enumerate() {
                let (lo, hi) = (request[0] as usize, request[1] as usize);
                if lo > hi || hi > resolved.token_ids.len() || hi > resolved.position_ids.len() {
                    return Err(refused(format!(
                        "instance {id} resolves a request over rows {lo}..{hi} of {} token(s)",
                        resolved.token_ids.len()
                    )));
                }
                tokens.extend_from_slice(&resolved.token_ids[lo..hi]);
                positions.extend_from_slice(&resolved.position_ids[lo..hi]);
                qo.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
                // The pages a decode attends are its own history, and the
                // last position it writes says how much of that there is.
                // Taken from the POSITIONS rather than from `kv_len`, because
                // a row writes where its position says and reading one page
                // fewer than it writes is an attention over a page this fire
                // itself filled -- and, on this driver, a paged append into a
                // page the plan did not name, which `validate_kv_writes`
                // refuses a moment later with a number nobody can trace back.
                let last = resolved.position_ids[lo..hi].iter().copied().max();
                let live = last.map_or(0, |p| pages_for(p.saturating_add(1), page));
                for logical in 0..live {
                    pages.push(physical(segment, logical)?);
                }
                page_indptr.push(u32::try_from(pages.len()).unwrap_or(u32::MAX));
                let derived = last.map_or(0, |p| driver::last_page_len(p.saturating_add(1), page));
                // The `kv_len` PORT, honoured by being CHECKED.
                //
                // `serve::load` claims `PIE_DECODE_ENVELOPE_PORTS`, which
                // names `EMBED_TOKENS`, `POSITIONS` and `KV_LEN`. The first
                // two are read straight off the resolution above. The third
                // is DERIVED here, from the position, for the reason the
                // comment above gives -- and a derivation is not a reading.
                // Where the two agree, nothing is lost and this costs a
                // comparison. Where they disagree, the guest asked for a
                // history of one length and this fire would attend another,
                // which is a silent answer to a question it asked out loud.
                //
                // They agree for every decode the engine builds today: a
                // guest's `kv_len.put(&next_length)` and its
                // `positions.put(&length)` come off the same counter. A guest
                // that parted them -- a sliding window, an evicted cache --
                // gets a refusal naming both numbers instead of an attention
                // over a span it did not ask for.
                if let Some(&stated) = resolved.kv_last_page_lens.get(which)
                    && stated != derived
                {
                    return Err(refused(format!(
                        "instance {id} resolves request {which} with a last page of {stated}, \
                         and its own positions say {derived}"
                    )));
                }
                lens.push(derived);
                // A device-resolved class names no read-out rows, which
                // `model_compiler::lower::Readouts::samples` takes as "the
                // last row of the request" -- what a decode means. The
                // boundary is still pushed: a CSR one short would give the
                // next request this one's span.
                sample_indptr.push(u32::try_from(samples.len()).unwrap_or(u32::MAX));
            }
            continue;
        }

        let (first, last) = member_requests(&sub.program_row_indptr, member, wire_rows)
            .ok_or_else(|| {
                refused(format!(
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
                return Err(refused(format!(
                    "request {r} spans rows {lo}..{hi} of {} token(s)",
                    sub.plan.token_ids.len()
                )));
            }
            tokens.extend_from_slice(&sub.plan.token_ids[lo..hi]);
            positions.extend_from_slice(&sub.plan.position_ids[lo..hi]);
            qo.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
            // Numbered from the REQUEST, and carried through unchanged. A row
            // past this request's own count would read a real distribution
            // belonging to another position, so it is refused here rather
            // than passed on to a lowering that would land it somewhere.
            for &named in sampling_rows(&sub.plan, r) {
                if named as usize >= hi - lo {
                    return Err(refused(format!(
                        "request {r} reads out row {named}, and has {} row(s)",
                        hi - lo
                    )));
                }
                samples.push(named);
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
                    // absent because the DEVICE classes state their pages in
                    // channels instead. So an empty CSR here is not a
                    // malformed plan on its own -- it is a member sent to the
                    // host path carrying no host geometry, which is what the
                    // class is named for. Naming the class is what makes that
                    // one step to find rather than four.
                    return Err(refused(format!(
                        "request {r} has no page span in a CSR of {} entries, and this member \
                         is geometry class {class}: a host-class member states its own pages, \
                         and only classes {} and {} state them elsewhere",
                        sub.plan.kv_page_indptr.len(),
                        driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                        driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY
                    )));
                }
            };
            if phi > sub.plan.kv_page_indices.len() {
                return Err(refused(format!(
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
    out.qo_indptr = qo;
    out.kv_page_indices = pages;
    out.kv_page_indptr = page_indptr;
    out.kv_last_page_lens = lens;
    out.sampling_indices = samples;
    out.sampling_indptr = sample_indptr;
    // `kv_len` is the plan's OTHER statement of the same span, and the two
    // have to be rebuilt together: `wire.rs` fills it from the page count and
    // the last page's length, and a decode envelope arrives with a zero there
    // beside pages this pass has just invented. Left alone it would say a
    // filled request attends nothing.
    out.kv_len = out
        .kv_page_indptr
        .windows(2)
        .zip(&out.kv_last_page_lens)
        .map(|(span, &last)| {
            let count = span[1].saturating_sub(span[0]);
            if count == 0 {
                0
            } else {
                (count - 1) * page + last
            }
        })
        .collect();
    Ok(Filled::Ready {
        plan: Box::new(out),
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

    /// One decode-envelope step: one member, no geometry on the wire.
    ///
    /// The shape the engine really sends -- `DecodeEnvelope::template` builds
    /// a placeholder token per row and leaves the page tables to the driver,
    /// which is why `token_ids` here is a zero and not the token that gets
    /// fired.
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

    /// A registry holding instance 7, an envelope program, and whichever of
    /// its three channels the caller seeded.
    fn bound(seeds: &[(u64, u32)]) -> crate::channel::Registry {
        let registration = envelope_program();
        let mut registry = crate::channel::Registry::new();
        let id = registry
            .register_program(registration.program_hash, registration.launch, Vec::new())
            .expect("a well-formed package");
        for channel in 0..3u64 {
            registry
                .register_channel(channel_spec(channel))
                .expect("a u32 ring");
        }
        let seeds: Vec<(u64, Vec<u8>)> = seeds
            .iter()
            .map(|&(channel, value)| (channel, value.to_le_bytes().to_vec()))
            .collect();
        registry
            .bind_instance(
                id,
                Some(7),
                crate::channel::Geometry::from_wire(driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE)
                    .expect("a class the registry serves"),
                &[0, 1, 2],
                &seeds,
            )
            .expect("instance 7");
        registry
    }

    fn channel_spec(id: u64) -> crate::channel::ChannelSpec {
        let plan = u32_ring(id);
        crate::channel::ChannelSpec {
            id: plan.channel_id,
            dtype: plan.dtype,
            shape: plan.shape.clone(),
            capacity: plan.capacity,
            role: crate::channel::HostRole::from_wire(plan.host_role),
            seeded: plan.seeded,
            direction: crate::channel::Direction::from_wire(plan.extern_dir),
            extern_name: plan.extern_name.clone(),
        }
    }

    /// The pages a fire names are its WORKING SET's, and the frame says where
    /// each one is.
    ///
    /// # The bug this pins
    ///
    /// Every conversation's first page is page 0 of its own working set, and
    /// they are not the same page of the pool. This driver applied no
    /// translation at all until this module existed, which gives every
    /// conversation pool page 0: the second one reads the first one's keys
    /// and answers fluently in its context, and no single-request test can
    /// see it because one conversation's identity map is consistent with
    /// itself.
    #[test]
    fn a_fires_pages_are_translated_to_the_ones_the_frame_placed() {
        let sub = wire();
        let frame = frame(vec![3, 2, 1, 0], sub.clone());
        let Filled::Ready { plan } = fill(
            &crate::channel::Registry::new(),
            &frame,
            &frame.steps[0],
            16,
        )
        .expect("a well-formed step") else {
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

    /// A frame that places nothing is naming physical pages already.
    #[test]
    fn a_frame_that_places_no_pages_leaves_them_alone() {
        let frame = frame(Vec::new(), wire());
        let Filled::Ready { plan } = fill(
            &crate::channel::Registry::new(),
            &frame,
            &frame.steps[0],
            16,
        )
        .expect("a well-formed step") else {
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
        let error = fill(
            &crate::channel::Registry::new(),
            &frame,
            &frame.steps[0],
            16,
        )
        .expect_err("page 9 of a two-page placement");
        assert!(
            format!("{error}").contains("working-set page 9"),
            "the refusal names the page: {error}"
        );
    }

    /// A step that says something about its members and leaves one out is
    /// refused, because the alternative is serving it as a class nobody chose.
    #[test]
    fn a_member_in_no_sub_batch_is_refused_rather_than_called_host() {
        let mut sub = wire();
        sub.sub_batch_indptr = vec![0, 0];
        let frame = frame(vec![3, 2, 1, 0], sub);
        let error = fill(
            &crate::channel::Registry::new(),
            &frame,
            &frame.steps[0],
            16,
        )
        .expect_err("a member in no sub-batch has no class to be served as");
        let said = format!("{error}");
        assert!(
            said.contains("step member 0 is in no sub-batch"),
            "the refusal names the member: {said}"
        );
    }

    /// An ABSENT table still reads as `Host`, which is the wire contract: a
    /// step with no sub-batching has no other class it could be.
    #[test]
    fn a_step_with_no_sub_batch_table_at_all_is_host() {
        let mut sub = wire();
        sub.sub_batch_indptr = Vec::new();
        sub.sub_batch_class = Vec::new();
        let frame = frame(vec![3, 2, 1, 0], sub);
        let Filled::Ready { plan } = fill(
            &crate::channel::Registry::new(),
            &frame,
            &frame.steps[0],
            16,
        )
        .expect("an absent sub-batch table is a host step") else {
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
        let frame = frame(vec![3, 2, 1, 0], sub);
        let error = fill(
            &crate::channel::Registry::new(),
            &frame,
            &frame.steps[0],
            16,
        )
        .expect_err("a host member states its own pages");
        let said = format!("{error}");
        assert!(
            said.contains("geometry class 0") && said.contains("state them elsewhere"),
            "the refusal names the class and what the other classes do: {said}"
        );
    }

    /// A decode states none of its geometry on the wire, and all of it on its
    /// channels: the token it embeds, where that token sits, and -- through
    /// the position -- how much history it attends.
    #[test]
    fn a_decode_envelope_is_filled_from_the_channels_the_last_fire_wrote() {
        let registry = bound(&[(0, 4242), (1, 20), (2, 21)]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        let Filled::Ready { plan } =
            fill(&registry, &frame, &frame.steps[0], 16).expect("a resolvable step")
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
        assert_eq!(
            plan.kv_len,
            vec![21],
            "the plan's other statement of the same span is rebuilt with it"
        );
    }

    /// The write target this driver derives from a filled plan is the cell
    /// the token's own position names, and it is INSIDE the pages the fill
    /// just placed.
    ///
    /// The arithmetic is `serve::launch`'s, restated on the values `fill`
    /// produces, because the two are one contract with no type joining them:
    /// launch indexes `kv_page_indices[kv_page_indptr[r] + pos / page]`, so a
    /// fill that placed one page fewer than the position needs would index
    /// the NEXT request's first page -- a real page, belonging to another
    /// conversation, with no fault of any kind.
    #[test]
    fn the_pages_a_filled_decode_places_cover_the_cell_its_position_writes() {
        let registry = bound(&[(0, 4242), (1, 20), (2, 21)]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        let Filled::Ready { plan } =
            fill(&registry, &frame, &frame.steps[0], 16).expect("a resolvable step")
        else {
            panic!("every channel is seeded, so nothing is early");
        };
        let page = 16u32;
        let pos = plan.position_ids[0];
        let base = plan.kv_page_indptr[0] as usize;
        let virt = base + (pos / page) as usize;
        assert!(
            virt < plan.kv_page_indptr[1] as usize,
            "position {pos} writes page {} of a request holding {}",
            pos / page,
            plan.kv_page_indptr[1] - plan.kv_page_indptr[0]
        );
        assert_eq!(
            (plan.kv_page_indices[virt], pos % page),
            (2, 4),
            "physical page 2 -- the frame placed working-set page 1 there -- row 4"
        );
    }

    /// An unfilled descriptor channel is early, not broken.
    #[test]
    fn a_decode_whose_producer_has_not_run_is_early() {
        let registry = bound(&[(1, 20), (2, 21)]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        let filled = fill(&registry, &frame, &frame.steps[0], 16).expect("not an error");
        assert!(
            matches!(filled, Filled::Early { channel: 0 }),
            "the empty token channel is named: {filled:?}"
        );
    }

    /// A `kv_len` that disagrees with the positions is refused, and the
    /// refusal states both numbers.
    ///
    /// The port is CLAIMED, so it may not be quietly ignored: a guest that
    /// parted its length from its positions asked for a history of one length
    /// and would be given another.
    #[test]
    fn a_stated_kv_len_that_disagrees_with_the_positions_is_refused() {
        let registry = bound(&[(0, 4242), (1, 20), (2, 33)]);
        let frame = frame(vec![3, 2, 1, 0], envelope_step());
        let error = fill(&registry, &frame, &frame.steps[0], 16)
            .expect_err("33 tokens of history under a position that says 21");
        let said = format!("{error}");
        assert!(
            said.contains("last page of 1") && said.contains("positions say 5"),
            "the refusal states both numbers: {said}"
        );
    }

    /// The further class is refused BY NAME rather than served as the
    /// smaller one.
    ///
    /// A device-geometry program states its pages, its CSR, its dense mask
    /// and its per-row write descriptor. This driver consumes none of the
    /// last two, so serving it as an envelope would derive exactly the three
    /// things it traced in order not to have derived -- and the fire would
    /// run, and be wrong, and say nothing.
    #[test]
    fn a_device_geometry_member_is_refused_by_name_and_not_served_as_an_envelope() {
        let mut sub = envelope_step();
        sub.sub_batch_class = vec![driver_api::PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY];
        let frame = frame(vec![3, 2, 1, 0], sub);
        let error = fill(
            &bound(&[(0, 1), (1, 0), (2, 1)]),
            &frame,
            &frame.steps[0],
            16,
        )
        .expect_err("this driver claims the envelope's ports and no more");
        let said = format!("{error}");
        assert!(
            said.contains("device geometry") && said.contains("write descriptor"),
            "the refusal names the class and what it could not honour: {said}"
        );
    }

    /// The question `launch` asks before it decides how to drive a frame.
    #[test]
    fn a_step_is_device_resolved_when_any_member_of_it_is() {
        assert!(!resolves_on_device(&wire()));
        assert!(resolves_on_device(&envelope_step()));
        let mut orphan = wire();
        // A table that exists and covers nothing: unreadable, and the safe
        // answer is the one that costs pipelining rather than the one that
        // fires over a cell that is not there.
        orphan.sub_batch_indptr = vec![0, 0];
        assert!(resolves_on_device(&orphan));
    }
}
