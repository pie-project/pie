//! Composing a batch's channel tickets: what each member promises about the
//! rings it will touch.
//!
//! The last of `compose.cpp`. A launch arrives with, for every member, the head
//! and tail its channels stood at when the engine composed the batch — a CSR of
//! `(expected_head, expected_tail)` pairs partitioned by member. This module
//! turns that wire form into per-member tickets, refusing every shape that
//! cannot be a batch.
//!
//! # Why a pin is required rather than optional
//!
//! A ticket pins a ring end so a fire can be ordered against every other fire
//! that touches it. The rule the C++ enforces and this keeps: a member that
//! **takes** from a channel must pin its head, and one that **puts** must pin
//! its tail. Without the pin there is no answer to "did someone else consume
//! this cell first", and the fire would race rather than refuse — which is why
//! `pipeline::readiness` refuses an unpinned put at fire time as
//! [`Reason::Unpinned`](crate::pipeline::Reason). This module is the earlier of
//! the two checks: catching it at composition names the member and the channel,
//! where catching it at the fire only knows the ring.
//!
//! # What the program says and what the request says are two facts
//!
//! The C++ folds them into one `bool requires_input`:
//!
//! ```cpp
//! .requires_input = expected_head != kNoChannelTicket ||
//!                   program.plan.requires_channel_input(dense),
//! ```
//!
//! One of those is a property of the *request* — the caller pinned a head — and
//! the other is a property of the *program*, fixed when it was bound. Once
//! merged, a fire that will not run because its input is missing cannot say
//! whether the requirement came from the batch it is in or from the program it
//! is, and those have different remedies: recompose, or do not submit this
//! program without a producer. [`ChannelTicket`] keeps both and answers the
//! merged question with [`ChannelTicket::requires_input`].
//!
//! # What this reuses
//!
//! [`Effect`] is already the driver's answer to "what does this fire do to this
//! channel" — `take`, `put` and `requires_full`, derived once when the program
//! was bound. The C++ asks `program.plan.takes_channel(dense)`,
//! `puts_channel(dense)` and `requires_channel_input(dense)` at composition
//! time, which is the same three questions of the same table; there is no
//! reason for a second copy. [`Ticket`] likewise already carries the pinned
//! pair, and is the type the readiness check consumes, so a composed ticket
//! hands straight to `prepare` without a conversion that could reorder it.

use crate::pipeline::{Effect, NO_TICKET, Ticket};

/// One member's promise about one of its channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelTicket {
    /// The registered channel this dense index names.
    pub channel_id: u64,
    /// The member-local index of the channel, which is also its index into the
    /// program's effect table.
    pub dense: u32,
    /// The pinned ends, in the form the readiness check consumes.
    pub ticket: Ticket,
    /// The **request** pinned a head for this channel.
    pub pinned_head: bool,
    /// The **program** requires a cell on this channel to run at all.
    pub program_requires_input: bool,
}

impl ChannelTicket {
    /// Whether this fire needs a cell present on the channel — for either
    /// reason. The C++'s single `requires_input`, recovered from the two facts
    /// that make it up rather than replacing them.
    #[must_use]
    pub const fn requires_input(&self) -> bool {
        self.pinned_head || self.program_requires_input
    }
}

/// One member's channels and what its program does to them.
///
/// The two slices are parallel and the same length; a member whose program has
/// three channels has three ids and three effects. They are separate slices
/// because they come from different places — the ids from the instance binding,
/// the effects from the bound program — and pairing them is this module's job.
#[derive(Clone, Copy, Debug)]
pub struct MemberChannels<'a> {
    /// Registered channel ids, in dense order.
    pub channel_ids: &'a [u64],
    /// What the member's program does to each, in the same order.
    pub effects: &'a [Effect],
}

/// Why a batch's tickets are not a batch.
///
/// Every variant names the member, and where the fault is per-channel it names
/// the channel too. The C++ returns a bare `PIE_STATUS_INVALID_ARGUMENT` from
/// four of these five sites with no message at all — only the array-length
/// case prints anything — so a caller learns that some member's tickets were
/// wrong and nothing more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TicketRefused {
    /// The CSR does not have one more entry than there are members.
    Indptr {
        /// Entries the partition holds.
        len: usize,
        /// Members in the batch.
        members: usize,
    },
    /// The head and tail arrays disagree in length, so no pairing exists.
    PairLengths {
        /// Heads supplied.
        heads: usize,
        /// Tails supplied.
        tails: usize,
    },
    /// A member's span runs backwards.
    SpanDescends {
        /// The member at fault.
        member: usize,
        /// Where its span began.
        lo: usize,
        /// Where its span ended.
        hi: usize,
    },
    /// A member's span is not as long as its channel list.
    SpanLength {
        /// The member at fault.
        member: usize,
        /// Entries the span covers.
        span: usize,
        /// Channels the member has.
        channels: usize,
    },
    /// A member's span reaches past the ticket arrays.
    SpanPastEnd {
        /// The member at fault.
        member: usize,
        /// One past the span's last entry.
        hi: usize,
        /// Entries the arrays hold.
        len: usize,
    },
    /// The member's channel ids and its program's effects disagree in length.
    ChannelsAgainstEffects {
        /// The member at fault.
        member: usize,
        /// Channel ids supplied.
        ids: usize,
        /// Effects the program declares.
        effects: usize,
    },
    /// A channel the member takes from has no pinned head.
    TakeWithoutHead {
        /// The member at fault.
        member: usize,
        /// Its dense channel index.
        dense: u32,
    },
    /// A channel the member puts to has no pinned tail.
    PutWithoutTail {
        /// The member at fault.
        member: usize,
        /// Its dense channel index.
        dense: u32,
    },
}

/// Compose every member's tickets from the wire CSR.
///
/// `indptr` partitions `expected_head`/`expected_tail` by member and has one
/// more entry than there are members.
///
/// # Errors
///
/// [`TicketRefused`], naming the member and where possible the channel. Nothing
/// is composed: a batch is accepted whole or not at all, so a refusal cannot
/// leave a caller holding some members' tickets.
pub fn compose_tickets(
    indptr: &[usize],
    members: &[MemberChannels<'_>],
    expected_head: &[u64],
    expected_tail: &[u64],
) -> Result<Vec<Vec<ChannelTicket>>, TicketRefused> {
    if indptr.len() != members.len() + 1 {
        return Err(TicketRefused::Indptr {
            len: indptr.len(),
            members: members.len(),
        });
    }
    if expected_head.len() != expected_tail.len() {
        return Err(TicketRefused::PairLengths {
            heads: expected_head.len(),
            tails: expected_tail.len(),
        });
    }

    let mut out = Vec::with_capacity(members.len());
    for (member, channels) in members.iter().enumerate() {
        if channels.channel_ids.len() != channels.effects.len() {
            return Err(TicketRefused::ChannelsAgainstEffects {
                member,
                ids: channels.channel_ids.len(),
                effects: channels.effects.len(),
            });
        }
        let (lo, hi) = (indptr[member], indptr[member + 1]);
        if hi < lo {
            return Err(TicketRefused::SpanDescends { member, lo, hi });
        }
        if hi > expected_head.len() {
            return Err(TicketRefused::SpanPastEnd {
                member,
                hi,
                len: expected_head.len(),
            });
        }
        if hi - lo != channels.channel_ids.len() {
            return Err(TicketRefused::SpanLength {
                member,
                span: hi - lo,
                channels: channels.channel_ids.len(),
            });
        }

        let mut tickets = Vec::with_capacity(hi - lo);
        for (dense, (&channel_id, &effect)) in channels
            .channel_ids
            .iter()
            .zip(channels.effects)
            .enumerate()
        {
            let head = expected_head[lo + dense];
            let tail = expected_tail[lo + dense];
            let dense = u32::try_from(dense).unwrap_or(u32::MAX);
            if effect.take && head == NO_TICKET {
                return Err(TicketRefused::TakeWithoutHead { member, dense });
            }
            if effect.put && tail == NO_TICKET {
                return Err(TicketRefused::PutWithoutTail { member, dense });
            }
            tickets.push(ChannelTicket {
                channel_id,
                dense,
                ticket: Ticket {
                    expected_head: head,
                    expected_tail: tail,
                },
                pinned_head: head != NO_TICKET,
                program_requires_input: effect.requires_full,
            });
        }
        out.push(tickets);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn effect(take: bool, put: bool, requires_full: bool) -> Effect {
        Effect {
            requires_full,
            requires_empty: false,
            take,
            put,
            capacity: 4,
        }
    }

    fn member<'a>(ids: &'a [u64], effects: &'a [Effect]) -> MemberChannels<'a> {
        MemberChannels {
            channel_ids: ids,
            effects,
        }
    }

    #[test]
    fn a_take_without_a_pinned_head_is_refused_at_composition_not_at_the_fire() {
        // The fire would refuse it too, as `Reason::Unpinned`, but by then all
        // it knows is the ring. Here the member and the dense index are still
        // in hand.
        let ids = [7u64];
        let effects = [effect(true, false, false)];
        let err = compose_tickets(&[0, 1], &[member(&ids, &effects)], &[NO_TICKET], &[5])
            .expect_err("a take must pin its head");
        assert_eq!(
            err,
            TicketRefused::TakeWithoutHead {
                member: 0,
                dense: 0
            }
        );
    }

    #[test]
    fn a_put_without_a_pinned_tail_is_refused_because_it_could_not_be_ordered() {
        let ids = [7u64];
        let effects = [effect(false, true, false)];
        let err = compose_tickets(&[0, 1], &[member(&ids, &effects)], &[3], &[NO_TICKET])
            .expect_err("a put must pin its tail");
        assert_eq!(
            err,
            TicketRefused::PutWithoutTail {
                member: 0,
                dense: 0
            }
        );
    }

    #[test]
    fn the_two_reasons_a_channel_needs_input_stay_apart_and_still_answer_together() {
        // The defect this module exists for: the C++ ORs them into one bool at
        // composition, so nothing downstream can tell a batch that pinned a
        // head from a program that cannot run without one.
        let ids = [1u64, 2, 3];
        let effects = [
            effect(false, false, true),  // the program demands input
            effect(true, false, false),  // the request pins a head
            effect(false, false, false), // neither
        ];
        let composed = compose_tickets(
            &[0, 3],
            &[member(&ids, &effects)],
            &[NO_TICKET, 9, NO_TICKET],
            &[NO_TICKET, NO_TICKET, NO_TICKET],
        )
        .expect("well formed");
        let m = &composed[0];

        assert!(m[0].program_requires_input && !m[0].pinned_head);
        assert!(m[1].pinned_head && !m[1].program_requires_input);
        assert!(!m[2].pinned_head && !m[2].program_requires_input);

        // And the merged answer the C++ kept is still available.
        assert_eq!(
            m.iter()
                .map(ChannelTicket::requires_input)
                .collect::<Vec<_>>(),
            [true, true, false]
        );
    }

    #[test]
    fn a_span_that_runs_backwards_is_refused_before_its_length_is_computed() {
        // `hi - lo` on a descending span wraps, and a wrapped length compared
        // against a channel count is a comparison that can accidentally pass.
        let ids = [1u64];
        let effects = [effect(false, false, false)];
        let err = compose_tickets(&[5, 2], &[member(&ids, &effects)], &[0; 8], &[0; 8])
            .expect_err("descending span");
        assert_eq!(
            err,
            TicketRefused::SpanDescends {
                member: 0,
                lo: 5,
                hi: 2
            }
        );
    }

    #[test]
    fn a_span_longer_than_the_arrays_is_refused_naming_both_numbers() {
        let ids = [1u64, 2];
        let effects = [effect(false, false, false); 2];
        let err = compose_tickets(&[0, 2], &[member(&ids, &effects)], &[0], &[0])
            .expect_err("the span reaches past the arrays");
        assert_eq!(
            err,
            TicketRefused::SpanPastEnd {
                member: 0,
                hi: 2,
                len: 1
            }
        );
    }

    #[test]
    fn a_span_that_does_not_cover_the_members_channels_is_refused() {
        let ids = [1u64, 2, 3];
        let effects = [effect(false, false, false); 3];
        let err = compose_tickets(&[0, 2], &[member(&ids, &effects)], &[0; 4], &[0; 4])
            .expect_err("two tickets for three channels");
        assert_eq!(
            err,
            TicketRefused::SpanLength {
                member: 0,
                span: 2,
                channels: 3
            }
        );
    }

    #[test]
    fn the_partition_must_have_one_more_entry_than_there_are_members() {
        let ids = [1u64];
        let effects = [effect(false, false, false)];
        let m = [member(&ids, &effects), member(&ids, &effects)];
        let err = compose_tickets(&[0, 1], &m, &[0; 2], &[0; 2]).expect_err("short indptr");
        assert_eq!(err, TicketRefused::Indptr { len: 2, members: 2 });
    }

    #[test]
    fn heads_and_tails_that_disagree_in_length_have_no_pairing() {
        let ids = [1u64];
        let effects = [effect(false, false, false)];
        let err = compose_tickets(&[0, 1], &[member(&ids, &effects)], &[0; 2], &[0; 3])
            .expect_err("no pairing");
        assert_eq!(err, TicketRefused::PairLengths { heads: 2, tails: 3 });
    }

    #[test]
    fn every_member_reads_its_own_span_of_the_shared_arrays() {
        let a = [10u64, 11];
        let b = [20u64];
        let ea = [effect(true, false, false); 2];
        let eb = [effect(false, true, false)];
        let composed = compose_tickets(
            &[0, 2, 3],
            &[member(&a, &ea), member(&b, &eb)],
            &[1, 2, NO_TICKET],
            &[NO_TICKET, NO_TICKET, 7],
        )
        .expect("well formed");

        assert_eq!(composed.len(), 2);
        assert_eq!(composed[0][0].ticket.expected_head, 1);
        assert_eq!(composed[0][1].ticket.expected_head, 2);
        assert_eq!(composed[0][1].channel_id, 11);
        assert_eq!(composed[1][0].ticket.expected_tail, 7);
        assert_eq!(composed[1][0].channel_id, 20);
        assert_eq!(composed[1][0].dense, 0, "dense is member-local, not global");
    }

    #[test]
    fn a_member_whose_ids_and_effects_disagree_is_refused_rather_than_zipped_short() {
        // `zip` would silently walk the shorter of the two, composing tickets
        // for a prefix of the member's channels and leaving the rest unpinned.
        let ids = [1u64, 2, 3];
        let effects = [effect(false, false, false); 2];
        let err = compose_tickets(&[0, 3], &[member(&ids, &effects)], &[0; 4], &[0; 4])
            .expect_err("ids and effects disagree");
        assert_eq!(
            err,
            TicketRefused::ChannelsAgainstEffects {
                member: 0,
                ids: 3,
                effects: 2
            }
        );
    }
}
