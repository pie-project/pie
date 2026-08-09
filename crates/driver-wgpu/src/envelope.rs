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
    Ready(Box<LaunchPlan>),
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

/// The geometry class of one member of a step.
fn class_of(sub: &StepSubmission, member: usize) -> u32 {
    for (b, window) in sub.sub_batch_indptr.windows(2).enumerate() {
        if (window[0] as usize..window[1] as usize).contains(&member) {
            return sub
                .sub_batch_class
                .get(b)
                .copied()
                .unwrap_or(driver_api::PIE_GEOMETRY_CLASS_HOST);
        }
    }
    driver_api::PIE_GEOMETRY_CLASS_HOST
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

    for (member, &row) in sub.roster_rows.iter().enumerate() {
        let segment = translation(frame, row)?;
        if class_of(sub, member) == driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE {
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
                // The pages a decode attends are its own history, and the
                // last position it writes says how much of that there is.
                // Taken from the POSITIONS rather than from `kv_len`,
                // because a row writes where its position says and reading
                // one page fewer than it writes is an attention over a page
                // this fire itself filled.
                let last = resolved.position_ids[lo..hi].iter().copied().max();
                let live = last.map_or(0, |p| pages_for(p.saturating_add(1), page));
                for logical in 0..live {
                    pages.push(physical(segment, logical)?);
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
                if let Some(&stated) = resolved.kv_last_page_lens.get(which) {
                    if stated != derived {
                        return Err(Unlaunched::Malformed(format!(
                            "instance {id} resolves request {which} with a last page of \
                             {stated}, and its own positions say {derived}"
                        )));
                    }
                }
                lens.push(derived);
            }
            continue;
        }

        let (first, last) = member_requests(&sub.program_row_indptr, member, wire_rows);
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
            qo.push(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
            let (plo, phi) = match (
                sub.plan.kv_page_indptr.get(r),
                sub.plan.kv_page_indptr.get(r + 1),
            ) {
                (Some(&plo), Some(&phi)) if phi >= plo => (plo as usize, phi as usize),
                _ => {
                    return Err(Unlaunched::Malformed(format!(
                        "request {r} has no page span in a CSR of {} entries",
                        sub.plan.kv_page_indptr.len()
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
    out.qo_indptr = qo;
    out.kv_page_indices = pages;
    out.kv_page_indptr = page_indptr;
    out.kv_last_page_lens = lens;
    // A request that named SEVERAL read-out rows cannot be served, and this is
    // the last place that can tell -- the table is about to be dropped.
    //
    // `Serving::over` forces every row to sample and hands back one readout
    // PER TURN: `Step::readout_of[i]` is turn `i`'s LAST row, the only one
    // that has seen the whole prompt. A caller that named `n` rows through
    // `fwd.readout(..)` -- which is what a speculative verifier does, one row
    // per drafted token -- gets one back, and the program that reads them
    // faults four layers down instead:
    //
    //     driver published poison epoch 1
    //     program faulted: launch program cannot be interpreted:
    //     logits intrinsic row range exceeds the forward's readout rows
    //
    // which is a real `pie serve` running `cacheback-speculative-decoding`.
    // The rows EXIST -- every row sampled -- and what is missing is a mapping
    // from a request to its row SPAN. Until there is one, refusing by name at
    // admission beats answering with the wrong number of rows in it.
    //
    // Checked HERE and not in `frames::unserved_in`, which is where the rest
    // of this driver's refusals live, because this path clears the table two
    // lines below: by the time `prepare` runs there is nothing left to see.
    if let Some(most) = crate::frames::widest_readout(&sub.plan) {
        if most > 1 {
            return Err(Unlaunched::Unserved(
                "a request naming several readout rows: this driver reads out \
                 one row per request, the last one",
            ));
        }
    }
    // Every read-out row is this step's own, and `Serving::over` forces every
    // row to sample anyway; a sampling table indexed against the WIRE's rows
    // would name rows this plan no longer has.
    out.sampling_indices = Vec::new();
    out.sampling_indptr = Vec::new();
    Ok(Filled::Ready(Box::new(out)))
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
        let Filled::Ready(plan) =
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

    /// A frame that places nothing is naming physical pages already.
    #[test]
    fn a_frame_that_places_no_pages_leaves_them_alone() {
        let frame = frame(Vec::new(), wire());
        let Filled::Ready(plan) =
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
        let Filled::Ready(plan) =
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
    /// this driver gives, and must still fill.
    #[test]
    fn a_wire_naming_several_readout_rows_is_refused_before_the_table_is_dropped() {
        let one = {
            let mut sub = wire();
            sub.plan.sampling_indices = vec![0];
            sub.plan.sampling_indptr = vec![0, 1];
            sub
        };
        let frame_one = frame(vec![3, 2, 1, 0], one.clone());
        assert!(
            matches!(
                fill(&Programs::default(), &frame_one, &one, 16),
                Ok(Filled::Ready(_))
            ),
            "one readout row per request is what this driver gives, and must fill"
        );

        let many = {
            let mut sub = wire();
            sub.plan.sampling_indices = vec![0, 1, 2];
            sub.plan.sampling_indptr = vec![0, 3];
            sub
        };
        let frame_many = frame(vec![3, 2, 1, 0], many.clone());
        let refused = fill(&Programs::default(), &frame_many, &many, 16)
            .expect_err("three readout rows for one request is not served silently");
        let said = refused.to_string();
        assert!(
            said.contains("readout rows"),
            "the refusal names what it could not do: {said}"
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
                Ok(Filled::Ready(_))
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
                Ok(Filled::Ready(_))
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
