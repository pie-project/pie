//! Admitting a fleet: turning many members' descriptions into one step's
//! request plan, or saying which member cannot be in it.
//!
//! `forward.cpp`'s `run_paged_batch_forward` opens with four hundred lines that
//! decide, member by member, whether a description can join the fire — and for
//! each accepted one, derives the per-request spans the dispatch and the
//! read-walk then use without re-deriving. That decision is a function of the
//! descriptions, the pool's geometry and the decoder's slot count. It needs no
//! device, and this is it.
//!
//! # A fleet is admitted per member, not as a batch
//!
//! One malformed member does not sink the fire. The C++ keeps parallel
//! `success[]` and `errors[]` arrays and `continue`s, so the members that are
//! well-formed still run — which is right: a fleet is other people's requests,
//! and one bad description is not a reason to fail everyone else's. [`Fleet`]
//! keeps that, as an accepted list and a rejection per member.
//!
//! What changes is that a rejection is a [`MemberRejected`] rather than a
//! string. The C++ builds eleven distinct prose messages and stores them in a
//! `std::vector<std::string>`, so the caller's only options are to log it or to
//! match on the text.
//!
//! # The slot every request writes must be its own
//!
//! Two requests in one fire writing one recurrent-state slot is not a race the
//! hardware resolves — it is the second one computing on top of the first's
//! state. The C++ tracks this with a `slot_owner` map and rejects the later
//! member, and the comment beside it records what the absence of the check
//! cost: *"every member fell to slot zero, the fire computed each sequence on
//! top of the last one's state, and the answers came back confident and
//! wrong."* [`marshal_fleet`] keeps the check and names both members.

use super::member::ForwardDesc;

/// The pool geometry a fleet is admitted against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolFacts {
    /// Physical pages the paged KV pool holds.
    pub total_pages: u32,
    /// Token rows per page.
    pub page_size: u32,
    /// Recurrent-state slots the decoder has; zero for a model without any.
    pub rs_slots: u32,
}

/// One request's spans into its member's arrays, derived once.
///
/// The C++ recomputes these inside a local `RequestSpan` and then again in the
/// dispatch walk; here they are produced once and carried. Half-open in every
/// axis, as the CSRs are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestPlan {
    /// Token range `[q0, q1)` into the member's `token_ids`/`position_ids`.
    pub q0: u32,
    /// One past the last token.
    pub q1: u32,
    /// Page range `[k0, k1)` into the member's `kv_pages`.
    pub k0: u32,
    /// One past the last page.
    pub k1: u32,
    /// Readout range `[s0, s1)` into the member's `readout_local_indices`.
    pub s0: u32,
    /// One past the last readout row.
    pub s1: u32,
    /// Rows used in this request's last page.
    pub last_page_len: u32,
    /// The recurrent-state slot this request reads and writes.
    pub slot: u32,
    /// Zero the slot before use: a fresh sequence.
    pub reset: bool,
    /// Read the slot's existing state.
    pub read: bool,
    /// Write the slot back.
    pub write: bool,
}

impl RequestPlan {
    /// The KV length this request addresses after its tokens are appended:
    /// `(pages - 1) * page_size + last_page_len`.
    #[must_use]
    pub const fn extent(&self, page_size: u32) -> u64 {
        (self.k1 - self.k0 - 1) as u64 * page_size as u64 + self.last_page_len as u64
    }
}

/// Why one member cannot join the fire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemberRejected {
    /// No tokens, or the token and position arrays disagree.
    TokensAgainstPositions {
        /// Tokens supplied.
        tokens: usize,
        /// Positions supplied.
        positions: usize,
    },
    /// A CSR does not close over the array it partitions.
    MalformedCsr {
        /// Which CSR, for the message.
        which: &'static str,
    },
    /// The decoder carries recurrent state and the member named no slot.
    ///
    /// The one the C++'s comment says used to be accepted, with every member
    /// falling to slot zero.
    NoRecurrentSlot,
    /// The per-request recurrent bindings do not cover the request count.
    RecurrentBindings {
        /// Requests the member has.
        requests: usize,
        /// Entries the shortest binding vector holds.
        bindings: usize,
    },
    /// A request's span runs backwards or past its array.
    InvalidSpan {
        /// The member-local request index.
        request: u32,
    },
    /// A request names a slot the decoder does not have.
    SlotOutOfRange {
        /// The member-local request index.
        request: u32,
        /// The slot named.
        slot: u32,
        /// Slots the decoder has.
        rs_slots: u32,
    },
    /// A request's last page is empty or over-full.
    LastPageLen {
        /// The member-local request index.
        request: u32,
        /// The length claimed.
        len: u32,
        /// Rows a page holds.
        page_size: u32,
    },
    /// A token's position is outside the KV extent its request addresses.
    PositionOutsideExtent {
        /// The member-local request index.
        request: u32,
        /// The offending position.
        position: u32,
        /// The extent it had to be under.
        extent: u64,
    },
    /// A page id is outside the pool.
    PageOutOfRange {
        /// The member-local request index.
        request: u32,
        /// The page named.
        page: u32,
        /// Pages the pool holds.
        total_pages: u32,
    },
    /// A readout index points past its own request's token span.
    ReadoutPastSpan {
        /// The member-local request index.
        request: u32,
        /// The offending readout index, request-local.
        readout: u32,
        /// Tokens the request has.
        query_count: u32,
    },
    /// Another member in this fire already writes that slot.
    SlotAlreadyOwned {
        /// The member-local request index.
        request: u32,
        /// The contested slot.
        slot: u32,
        /// The member that claimed it first.
        owner: usize,
    },
}

/// One member's outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Accepted, with its per-request plans.
    Accepted(Vec<RequestPlan>),
    /// Rejected, with the reason.
    Rejected(MemberRejected),
}

/// The fire's admission outcome, one entry per member in the order given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fleet {
    /// Per-member outcomes, parallel to the descriptions.
    pub members: Vec<Admission>,
}

impl Fleet {
    /// Whether any member was accepted. A fire with none has nothing to do.
    #[must_use]
    pub fn any_accepted(&self) -> bool {
        self.members
            .iter()
            .any(|m| matches!(m, Admission::Accepted(_)))
    }
}

/// Admit a fleet: plan every member that can join, and say why each other
/// cannot.
///
/// Members are considered in order, and slot ownership is claimed in that
/// order, so an earlier member keeps a contested slot. That is the C++'s
/// behaviour and it is the right one — admission is not the place to decide
/// whose request matters more, and rejecting the later member is the outcome a
/// caller can retry.
#[must_use]
pub fn marshal_fleet(descs: &[ForwardDesc], facts: PoolFacts) -> Fleet {
    let mut slot_owner: Vec<(u32, usize)> = Vec::new();
    let mut members = Vec::with_capacity(descs.len());

    for (index, desc) in descs.iter().enumerate() {
        match plan_member(desc, facts) {
            Err(reason) => members.push(Admission::Rejected(reason)),
            Ok(plans) => {
                // Claim every slot this member writes, or reject it whole: a
                // member half-admitted would have some of its requests planned
                // against slots another member also writes.
                let clash = plans.iter().enumerate().find_map(|(request, plan)| {
                    plan.write
                        .then(|| {
                            slot_owner
                                .iter()
                                .find(|(slot, _)| *slot == plan.slot)
                                .map(|&(slot, owner)| (request, slot, owner))
                        })
                        .flatten()
                });
                if let Some((request, slot, owner)) = clash {
                    members.push(Admission::Rejected(MemberRejected::SlotAlreadyOwned {
                        request: u32::try_from(request).unwrap_or(u32::MAX),
                        slot,
                        owner,
                    }));
                    continue;
                }
                for plan in plans.iter().filter(|p| p.write) {
                    slot_owner.push((plan.slot, index));
                }
                members.push(Admission::Accepted(plans));
            }
        }
    }

    Fleet { members }
}

/// Validate one member and derive its request plans.
///
/// # Errors
///
/// [`MemberRejected`], naming the request where the fault is per-request.
pub fn plan_member(
    desc: &ForwardDesc,
    facts: PoolFacts,
) -> Result<Vec<RequestPlan>, MemberRejected> {
    if desc.token_ids.is_empty() || desc.token_ids.len() != desc.position_ids.len() {
        return Err(MemberRejected::TokensAgainstPositions {
            tokens: desc.token_ids.len(),
            positions: desc.position_ids.len(),
        });
    }
    let requests = desc
        .qo_indptr
        .len()
        .checked_sub(1)
        .ok_or(MemberRejected::MalformedCsr { which: "qo_indptr" })?;
    if requests == 0
        || desc.qo_indptr.first() != Some(&0)
        || desc.qo_indptr.last().copied() != u32::try_from(desc.token_ids.len()).ok()
    {
        return Err(MemberRejected::MalformedCsr { which: "qo_indptr" });
    }
    if desc.kv_page_indptr.len() != desc.qo_indptr.len()
        || desc.kv_page_indptr.first() != Some(&0)
        || desc.kv_page_indptr.last().copied() != u32::try_from(desc.kv_pages.len()).ok()
    {
        return Err(MemberRejected::MalformedCsr {
            which: "kv_page_indptr",
        });
    }
    if desc.kv_last_page_lens.len() != requests {
        return Err(MemberRejected::MalformedCsr {
            which: "kv_last_page_lens",
        });
    }
    if desc.sampling_indptr.len() != desc.qo_indptr.len()
        || desc.sampling_indptr.first() != Some(&0)
        || desc.sampling_indptr.last().copied()
            != u32::try_from(desc.readout_local_indices.len()).ok()
    {
        return Err(MemberRejected::MalformedCsr {
            which: "sampling_indptr",
        });
    }

    // The decoder's state has to live somewhere the member names. The C++
    // records what accepting this used to cost: every member fell to slot zero
    // and each sequence computed on top of the last one's state.
    //
    // The C++ then carries a fallback — `slot = accepted_requests.size() +
    // request` for a member with no slot — which is UNREACHABLE. This gate
    // rejects `!has_rs_slot` whenever `rs_slots > 0`, and when `rs_slots == 0`
    // the range check below rejects every synthesized slot, since any unsigned
    // value is `>= 0`. It reads like a policy ("members without slots get
    // positional ones") and is dead code; there is no such policy here.
    let bindings = desc
        .request_rs_slot_ids
        .len()
        .min(desc.request_rs_reset.len())
        .min(desc.request_rs_read.len())
        .min(desc.request_rs_write.len());
    if facts.rs_slots > 0 && bindings == 0 {
        return Err(MemberRejected::NoRecurrentSlot);
    }
    if bindings > 0 && bindings != requests {
        return Err(MemberRejected::RecurrentBindings { requests, bindings });
    }

    let mut plans = Vec::with_capacity(requests);
    for request in 0..requests {
        let r = u32::try_from(request).unwrap_or(u32::MAX);
        let plan = RequestPlan {
            q0: desc.qo_indptr[request],
            q1: desc.qo_indptr[request + 1],
            k0: desc.kv_page_indptr[request],
            k1: desc.kv_page_indptr[request + 1],
            s0: desc.sampling_indptr[request],
            s1: desc.sampling_indptr[request + 1],
            last_page_len: desc.kv_last_page_lens[request],
            slot: desc.request_rs_slot_ids.get(request).copied().unwrap_or(0),
            reset: desc.request_rs_reset.get(request).copied().unwrap_or(false),
            read: desc.request_rs_read.get(request).copied().unwrap_or(false),
            write: desc.request_rs_write.get(request).copied().unwrap_or(false),
        };

        let tokens = u32::try_from(desc.token_ids.len()).unwrap_or(u32::MAX);
        let pages = u32::try_from(desc.kv_pages.len()).unwrap_or(u32::MAX);
        let readouts = u32::try_from(desc.readout_local_indices.len()).unwrap_or(u32::MAX);
        if plan.q1 <= plan.q0
            || plan.q1 > tokens
            || plan.k1 <= plan.k0
            || plan.k1 > pages
            || plan.s1 < plan.s0
            || plan.s1 > readouts
        {
            return Err(MemberRejected::InvalidSpan { request: r });
        }
        if facts.rs_slots > 0 && plan.slot >= facts.rs_slots {
            return Err(MemberRejected::SlotOutOfRange {
                request: r,
                slot: plan.slot,
                rs_slots: facts.rs_slots,
            });
        }
        if plan.last_page_len == 0 || plan.last_page_len > facts.page_size {
            return Err(MemberRejected::LastPageLen {
                request: r,
                len: plan.last_page_len,
                page_size: facts.page_size,
            });
        }
        let extent = plan.extent(facts.page_size);
        if let Some(&position) = desc.position_ids[plan.q0 as usize..plan.q1 as usize]
            .iter()
            .find(|&&p| u64::from(p) >= extent)
        {
            return Err(MemberRejected::PositionOutsideExtent {
                request: r,
                position,
                extent,
            });
        }
        if let Some(&page) = desc.kv_pages[plan.k0 as usize..plan.k1 as usize]
            .iter()
            .find(|&&p| p >= facts.total_pages)
        {
            return Err(MemberRejected::PageOutOfRange {
                request: r,
                page,
                total_pages: facts.total_pages,
            });
        }
        // Every readout row must land inside its own request's tokens. The
        // concatenation below rebases these to fleet rows by adding the
        // request's token base, and this is what makes that addition safe --
        // which is why `concat_fleet` needs no bound of its own.
        let query_count = plan.q1 - plan.q0;
        if let Some(&readout) = desc.readout_local_indices[plan.s0 as usize..plan.s1 as usize]
            .iter()
            .find(|&&r| r >= query_count)
        {
            return Err(MemberRejected::ReadoutPastSpan {
                request: r,
                readout,
                query_count,
            });
        }
        plans.push(plan);
    }
    Ok(plans)
}

/// One step's concatenated inputs: every accepted member's arrays laid end to
/// end, with each member's request-local indices rebased onto the fire.
///
/// The C++'s `BatchStepInputs`, filled. What it is *for* is that the kernels
/// see one batch: a fleet of five conversations is one token array, one page
/// array and one CSR over both, so nothing downstream has to know a member
/// existed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepInputs {
    /// Every accepted request's tokens, in fleet order.
    pub token_ids: Vec<u32>,
    /// Their positions, parallel to `token_ids`.
    pub position_ids: Vec<u32>,
    /// Token CSR over the fleet's requests; closed, so `requests + 1` long.
    pub qo_indptr: Vec<u32>,
    /// Page CSR over the fleet's requests; closed the same way.
    pub kv_page_indptr: Vec<u32>,
    /// Every accepted request's pages, in fleet order.
    pub kv_page_indices: Vec<u32>,
    /// Rows used in each request's last page.
    pub kv_last_page_lens: Vec<u32>,
    /// Each request's recurrent-state slot.
    pub rs_slot_ids: Vec<u32>,
    /// Each request's flags byte; bit 0 is reset.
    pub rs_slot_flags: Vec<u8>,
    /// Per token: the page its K/V is written into.
    pub w_page: Vec<u32>,
    /// Per token: the row within that page.
    pub w_off: Vec<u32>,
    /// Dense mask rows, `attention_mask_stride` bytes each, or empty.
    pub attention_mask: Vec<u8>,
    /// Per token: whether that row's mask is meaningful. Empty with the mask.
    pub attention_mask_enabled: Vec<u8>,
    /// Bytes per mask row.
    pub attention_mask_stride: u32,
    /// Per token: whether anything reads that row's logits.
    pub row_needs_logits: Vec<u8>,
}

/// Where one member's readout rows landed in the fleet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberRows {
    /// The member's index in the input slice.
    pub member: usize,
    /// Its readout rows, as fleet-global token rows.
    pub rows: Vec<u32>,
}

/// Lay every accepted member end to end.
///
/// `mask_stride` is the dense mask's pitch in bytes, a property of the
/// decoder's geometry rather than of any member.
///
/// # Why the mask arrays are usually empty
///
/// A dense mask is one byte per addressable KV token per row. Materialising it
/// costs `rows * stride` of zero-fill and the same again copying it to the IO
/// slot — and because a non-empty `attention_mask_enabled` is what tells the
/// step "this batch is masked", pushing a zero per token made **every** batch
/// pay both for a buffer no kernel reads. The C++ records the bill: 8.4 MB of
/// memory traffic per step at 32 lanes, growing linearly with lane count. So
/// the arrays stay empty unless some member actually carries a mask, and that
/// decision is taken once for the fleet rather than per member.
#[must_use]
pub fn concat_fleet(
    descs: &[ForwardDesc],
    fleet: &Fleet,
    facts: PoolFacts,
    mask_stride: u32,
) -> (StepInputs, Vec<MemberRows>) {
    let accepted = || {
        descs
            .iter()
            .zip(&fleet.members)
            .enumerate()
            .filter_map(|(i, (d, a))| match a {
                Admission::Accepted(plans) => Some((i, d, plans)),
                Admission::Rejected(_) => None,
            })
    };
    let masked = accepted().any(|(_, d, _)| !d.attention_mask.is_empty());

    let mut step = StepInputs {
        attention_mask_stride: mask_stride,
        ..StepInputs::default()
    };
    let mut per_member = Vec::new();

    for (index, desc, plans) in accepted() {
        let mut rows = Vec::new();
        for plan in plans {
            let token_base = u32::try_from(step.token_ids.len()).unwrap_or(u32::MAX);
            step.qo_indptr.push(token_base);
            step.kv_page_indptr
                .push(u32::try_from(step.kv_page_indices.len()).unwrap_or(u32::MAX));

            let (q0, q1) = (plan.q0 as usize, plan.q1 as usize);
            let (k0, k1) = (plan.k0 as usize, plan.k1 as usize);
            step.token_ids.extend_from_slice(&desc.token_ids[q0..q1]);
            step.position_ids
                .extend_from_slice(&desc.position_ids[q0..q1]);
            step.kv_page_indices
                .extend_from_slice(&desc.kv_pages[k0..k1]);
            step.kv_last_page_lens.push(plan.last_page_len);
            step.rs_slot_ids.push(plan.slot);
            step.rs_slot_flags.push(u8::from(plan.reset));

            for token in q0..q1 {
                let position = desc.position_ids[token];
                if desc.has_write_desc {
                    step.w_page.push(desc.w_page[token]);
                    step.w_off.push(desc.w_off[token]);
                } else {
                    // Derived from the request's own page list. The index is in
                    // range because `plan_member` proved every position is under
                    // the request's extent: `position / page_size <= k1 - k0 - 1`
                    // follows from it, and nothing else does.
                    let page = k0 + (position / facts.page_size) as usize;
                    step.w_page.push(desc.kv_pages[page]);
                    step.w_off.push(position % facts.page_size);
                }
                if masked {
                    let has = !desc.attention_mask.is_empty();
                    step.attention_mask_enabled.push(u8::from(has));
                    let base = step.attention_mask.len();
                    step.attention_mask.resize(base + mask_stride as usize, 0);
                    if has {
                        let stride = desc.attention_mask_stride as usize;
                        let from = token * stride;
                        let take = stride.min(mask_stride as usize);
                        step.attention_mask[base..base + take]
                            .copy_from_slice(&desc.attention_mask[from..from + take]);
                    }
                }
            }

            // Request-local readout indices become fleet rows. Safe without a
            // bound because `plan_member` refused any index past its request's
            // token span -- see `MemberRejected::ReadoutPastSpan`.
            let (s0, s1) = (plan.s0 as usize, plan.s1 as usize);
            rows.extend(
                desc.readout_local_indices[s0..s1]
                    .iter()
                    .map(|&local| token_base + local),
            );
        }
        per_member.push(MemberRows {
            member: index,
            rows,
        });
    }

    // Close both CSRs. A partition of N spans has N+1 entries and the last one
    // is the total; the C++ pushes these after the loop for the same reason.
    step.qo_indptr
        .push(u32::try_from(step.token_ids.len()).unwrap_or(u32::MAX));
    step.kv_page_indptr
        .push(u32::try_from(step.kv_page_indices.len()).unwrap_or(u32::MAX));

    step.row_needs_logits = vec![0; step.token_ids.len()];
    for member in &per_member {
        for &row in &member.rows {
            step.row_needs_logits[row as usize] = 1;
        }
    }

    (step, per_member)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> PoolFacts {
        PoolFacts {
            total_pages: 64,
            page_size: 16,
            rs_slots: 4,
        }
    }

    /// A one-request member: `tokens` tokens, `pages` pages, last page full.
    fn member(tokens: u32, pages: u32, slot: u32) -> ForwardDesc {
        ForwardDesc {
            token_ids: (0..tokens).collect(),
            position_ids: (0..tokens).collect(),
            kv_pages: (0..pages).collect(),
            qo_indptr: vec![0, tokens],
            kv_page_indptr: vec![0, pages],
            kv_last_page_lens: vec![16],
            sampling_indptr: vec![0, 1],
            readout_local_indices: vec![tokens - 1],
            request_rs_slot_ids: vec![slot],
            request_rs_reset: vec![false],
            request_rs_read: vec![true],
            request_rs_write: vec![true],
            ..ForwardDesc::default()
        }
    }

    #[test]
    fn a_well_formed_member_plans_its_request_once() {
        let plans = plan_member(&member(16, 1, 2), facts()).expect("well formed");
        assert_eq!(plans.len(), 1);
        let p = plans[0];
        assert_eq!((p.q0, p.q1), (0, 16));
        assert_eq!((p.k0, p.k1), (0, 1));
        assert_eq!(p.slot, 2);
        assert_eq!(p.extent(16), 16, "one page, sixteen rows used");
    }

    #[test]
    fn two_members_writing_one_slot_are_not_both_admitted() {
        // The defect the C++'s comment records: without this, the second
        // sequence computes on top of the first's recurrent state and the
        // answers come back confident and wrong.
        let fleet = marshal_fleet(&[member(16, 1, 1), member(16, 1, 1)], facts());
        assert!(matches!(fleet.members[0], Admission::Accepted(_)));
        assert_eq!(
            fleet.members[1],
            Admission::Rejected(MemberRejected::SlotAlreadyOwned {
                request: 0,
                slot: 1,
                owner: 0
            })
        );
        assert!(fleet.any_accepted());
    }

    #[test]
    fn the_earlier_member_keeps_a_contested_slot() {
        let fleet = marshal_fleet(
            &[member(16, 1, 3), member(8, 1, 3), member(16, 1, 0)],
            facts(),
        );
        assert!(matches!(fleet.members[0], Admission::Accepted(_)));
        assert!(matches!(fleet.members[1], Admission::Rejected(_)));
        assert!(
            matches!(fleet.members[2], Admission::Accepted(_)),
            "a rejection must not sink the members after it"
        );
    }

    #[test]
    fn one_malformed_member_does_not_sink_the_rest_of_the_fleet() {
        let mut bad = member(16, 1, 0);
        bad.position_ids.pop();
        let fleet = marshal_fleet(&[bad, member(16, 1, 1)], facts());
        assert!(matches!(
            fleet.members[0],
            Admission::Rejected(MemberRejected::TokensAgainstPositions { .. })
        ));
        assert!(matches!(fleet.members[1], Admission::Accepted(_)));
    }

    #[test]
    fn a_decoder_with_recurrent_state_refuses_a_member_that_names_no_slot() {
        let mut d = member(16, 1, 0);
        d.request_rs_slot_ids.clear();
        d.request_rs_reset.clear();
        d.request_rs_read.clear();
        d.request_rs_write.clear();
        assert_eq!(
            plan_member(&d, facts()),
            Err(MemberRejected::NoRecurrentSlot)
        );
    }

    #[test]
    fn a_position_past_the_requests_paged_extent_is_refused_naming_both() {
        // Two pages, last one holding three rows: the extent is 16 + 3 = 19,
        // so position 19 is one past the end.
        let mut d = member(20, 2, 0);
        d.kv_last_page_lens = vec![3];
        d.position_ids = (0..20).collect();
        let err = plan_member(&d, facts()).expect_err("position 19 is outside");
        assert_eq!(
            err,
            MemberRejected::PositionOutsideExtent {
                request: 0,
                position: 19,
                extent: 19
            }
        );
    }

    #[test]
    fn a_last_page_length_of_zero_is_refused_rather_than_making_the_extent_a_page_short() {
        let mut d = member(16, 1, 0);
        d.kv_last_page_lens = vec![0];
        assert_eq!(
            plan_member(&d, facts()),
            Err(MemberRejected::LastPageLen {
                request: 0,
                len: 0,
                page_size: 16
            })
        );
    }

    #[test]
    fn a_page_id_outside_the_pool_is_refused_naming_the_bound() {
        let mut d = member(16, 1, 0);
        d.kv_pages = vec![99];
        assert_eq!(
            plan_member(&d, facts()),
            Err(MemberRejected::PageOutOfRange {
                request: 0,
                page: 99,
                total_pages: 64
            })
        );
    }

    #[test]
    fn a_slot_outside_the_decoders_range_is_refused() {
        assert_eq!(
            plan_member(&member(16, 1, 9), facts()),
            Err(MemberRejected::SlotOutOfRange {
                request: 0,
                slot: 9,
                rs_slots: 4
            })
        );
    }

    #[test]
    fn recurrent_bindings_shorter_than_the_request_count_are_refused() {
        let mut d = member(16, 1, 0);
        d.qo_indptr = vec![0, 8, 16];
        d.kv_page_indptr = vec![0, 1, 1];
        d.kv_last_page_lens = vec![16, 16];
        d.sampling_indptr = vec![0, 1, 1];
        // Two requests, one binding.
        assert_eq!(
            plan_member(&d, facts()),
            Err(MemberRejected::RecurrentBindings {
                requests: 2,
                bindings: 1
            })
        );
    }

    #[test]
    fn a_csr_that_does_not_close_over_its_array_names_which_one() {
        let mut d = member(16, 1, 0);
        d.qo_indptr = vec![0, 15];
        assert_eq!(
            plan_member(&d, facts()),
            Err(MemberRejected::MalformedCsr { which: "qo_indptr" })
        );

        let mut d = member(16, 1, 0);
        d.sampling_indptr = vec![0, 5];
        assert_eq!(
            plan_member(&d, facts()),
            Err(MemberRejected::MalformedCsr {
                which: "sampling_indptr"
            })
        );
    }

    #[test]
    fn a_readout_past_its_own_requests_tokens_is_refused_so_the_rebase_needs_no_bound() {
        let mut d = member(16, 1, 0);
        d.readout_local_indices = vec![16];
        assert_eq!(
            plan_member(&d, facts()),
            Err(MemberRejected::ReadoutPastSpan {
                request: 0,
                readout: 16,
                query_count: 16
            })
        );
    }

    #[test]
    fn two_members_concatenate_into_one_batch_with_both_csrs_closed() {
        let descs = [member(4, 1, 0), member(6, 2, 1)];
        let fleet = marshal_fleet(&descs, facts());
        let (step, rows) = concat_fleet(&descs, &fleet, facts(), 8);

        assert_eq!(step.token_ids.len(), 10, "4 + 6 tokens");
        assert_eq!(step.kv_page_indices.len(), 3, "1 + 2 pages");
        assert_eq!(step.qo_indptr, [0, 4, 10], "two spans, closed");
        assert_eq!(step.kv_page_indptr, [0, 1, 3], "two spans, closed");
        assert_eq!(step.rs_slot_ids, [0, 1]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rows, [3], "member 0's last token, fleet row 3");
        assert_eq!(
            rows[1].rows,
            [9],
            "member 1's last token, rebased past member 0"
        );
    }

    #[test]
    fn a_rejected_member_contributes_nothing_and_does_not_shift_the_others() {
        let mut bad = member(4, 1, 0);
        bad.kv_pages = vec![99]; // out of the pool
        let descs = [bad, member(6, 1, 1)];
        let fleet = marshal_fleet(&descs, facts());
        let (step, rows) = concat_fleet(&descs, &fleet, facts(), 8);

        assert_eq!(step.token_ids.len(), 6, "only the accepted member");
        assert_eq!(step.qo_indptr, [0, 6]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].member, 1, "the surviving member keeps its index");
        assert_eq!(rows[0].rows, [5], "rebased from zero, not from four");
    }

    #[test]
    fn a_write_target_without_a_write_descriptor_is_derived_from_the_requests_own_pages() {
        // Page 7 then page 9; page_size 16. Token at position 20 lands in the
        // request's SECOND page (20 / 16 == 1) at row 4.
        let mut d = member(21, 2, 0);
        d.kv_pages = vec![7, 9];
        d.kv_last_page_lens = vec![5];
        d.position_ids = (0..21).collect();
        let descs = [d];
        let fleet = marshal_fleet(&descs, facts());
        assert!(matches!(fleet.members[0], Admission::Accepted(_)));
        let (step, _) = concat_fleet(&descs, &fleet, facts(), 8);

        assert_eq!(step.w_page[0], 7, "position 0 is in the first page");
        assert_eq!(step.w_off[0], 0);
        assert_eq!(step.w_page[20], 9, "position 20 is in the second");
        assert_eq!(step.w_off[20], 4);
    }

    #[test]
    fn only_the_readout_rows_need_logits_and_the_rest_never_project() {
        // The point of the byte: on a prefill this is one row out of N, and
        // the other N-1 never pay for the lm_head projection.
        let descs = [member(8, 1, 0)];
        let fleet = marshal_fleet(&descs, facts());
        let (step, _) = concat_fleet(&descs, &fleet, facts(), 8);

        assert_eq!(step.row_needs_logits.len(), 8);
        assert_eq!(step.row_needs_logits, [0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn an_unmasked_fleet_materialises_no_mask_at_all() {
        // The 8.4 MB per step the C++ records. A zero pushed per token would
        // make `attention_mask_enabled` non-empty, which is itself what tells
        // the step the batch is masked.
        let descs = [member(8, 1, 0), member(8, 1, 1)];
        let fleet = marshal_fleet(&descs, facts());
        let (step, _) = concat_fleet(&descs, &fleet, facts(), 4096);

        assert!(step.attention_mask.is_empty());
        assert!(step.attention_mask_enabled.is_empty());
    }

    #[test]
    fn one_masked_member_makes_the_fleet_masked_and_the_others_rows_are_zero() {
        let mut masked = member(2, 1, 0);
        masked.attention_mask = vec![1, 1, 1, 1, 2, 2, 2, 2];
        masked.attention_mask_stride = 4;
        let descs = [masked, member(2, 1, 1)];
        let fleet = marshal_fleet(&descs, facts());
        let (step, _) = concat_fleet(&descs, &fleet, facts(), 4);

        assert_eq!(step.attention_mask_enabled, [1, 1, 0, 0]);
        assert_eq!(step.attention_mask.len(), 16, "four rows of four bytes");
        assert_eq!(&step.attention_mask[0..4], [1, 1, 1, 1]);
        assert_eq!(&step.attention_mask[4..8], [2, 2, 2, 2]);
        assert_eq!(&step.attention_mask[8..16], [0; 8], "the unmasked member");
    }

    #[test]
    fn a_member_that_writes_nothing_claims_no_slot() {
        // A read-only member cannot contest a slot, because contention is
        // about who writes it.
        let mut a = member(16, 1, 1);
        a.request_rs_write = vec![false];
        let fleet = marshal_fleet(&[a, member(16, 1, 1)], facts());
        assert!(matches!(fleet.members[0], Admission::Accepted(_)));
        assert!(
            matches!(fleet.members[1], Admission::Accepted(_)),
            "the writer may still take a slot only a reader touched"
        );
    }
}
