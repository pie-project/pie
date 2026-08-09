//! Whether a fire may run right now.
//!
//! A fire reads and writes channels, and a channel is a bounded ring shared
//! with whoever else touches it. Before anything is encoded the host asks, for
//! every channel the fire touches: is there something to take, is there room to
//! put, and is this ring still at the sequence the batch was composed against?
//! Three answers are possible and they are not interchangeable —
//!
//! * [`Readiness::Ready`] — encode it.
//! * [`Readiness::Retry`] — nothing is wrong; the fire is early. The producer
//!   has not run, or the consumer has not drained, or another fire moved the
//!   ring since this batch was composed. Try again next pass.
//! * [`Readiness::Failed`] — the fire will never be runnable. A poisoned or
//!   closed channel, or a ring whose words are not self-consistent.
//!
//! Collapsing the middle one into either of the others is the mistake this
//! module exists to prevent: reported as a failure it kills a request that was
//! merely early, and reported as ready it encodes a kernel that reads a cell
//! nobody wrote.
//!
//! # Tickets
//!
//! A ticket is the head and tail a channel had when the batch was composed. It
//! is not a readiness condition of its own — it is the check that the ring has
//! not moved underneath a decision already made. A fire composed against
//! `head = 4` that arrives to find `head = 5` has had its cell taken by someone
//! else, and running it would consume the wrong one. That is a retry, not a
//! failure: recomposing against the new head is exactly the remedy.
//!
//! A [`NO_TICKET`] entry means the composer did not pin that end, and the
//! corresponding check is skipped. The C++ spelled it `~std::uint64_t{0}` and
//! compared against it inline; here it is a named constant, because a sentinel
//! that shares a field with real values is only safe while every reader
//! remembers it is there.
//!
//! # The put credit
//!
//! An op that both takes from and puts to one channel does not need a free slot
//! before it runs: the take it performs first frees one. So a put is allowed
//! against a ring at exactly capacity when the same fire is also taking from it
//! and is waiting on that ring being non-empty. That one-slot credit is the
//! only reason a self-recursive stage does not deadlock at capacity, and it is
//! why [`Effect`] carries `take` and `put` separately rather than one direction.

use crate::pipeline::channel::ChannelState;

/// A ticket end the composer left unpinned.
///
/// Named because it shares a field with real sequence numbers: a ring whose
/// head genuinely reached `u64::MAX` would be indistinguishable, which at one
/// take per nanosecond is some six hundred years away and is the reason the
/// sentinel is acceptable rather than the reason it is safe.
pub const NO_TICKET: u64 = u64::MAX;

/// What one fire does to one channel.
///
/// Derived once when the program is bound, from the ops that touch the channel,
/// and read on every readiness check thereafter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Effect {
    /// The fire's first touch is a take or read, so it needs a cell to exist.
    pub requires_full: bool,
    /// The fire's first touch is a put, so it needs room.
    pub requires_empty: bool,
    /// The fire takes from this channel at some point.
    pub take: bool,
    /// The fire puts to this channel at some point.
    pub put: bool,
    /// How many live cells the ring holds.
    pub capacity: u32,
}

/// The head and tail a channel had when the batch was composed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket {
    /// The head to require, or [`NO_TICKET`].
    pub expected_head: u64,
    /// The tail to require, or [`NO_TICKET`]. A fire that puts must have one:
    /// see [`check`].
    pub expected_tail: u64,
}

impl Default for Ticket {
    /// Unpinned at both ends, which is what a composer that did not pin the
    /// channel produces.
    fn default() -> Self {
        Ticket {
            expected_head: NO_TICKET,
            expected_tail: NO_TICKET,
        }
    }
}

/// Why a fire is not ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The ring is poisoned: a producer faulted it.
    Poisoned,
    /// The ring is closed: no further cell will ever arrive.
    Closed,
    /// `tail < head`, which no sequence of takes and puts can produce.
    ///
    /// The C++ folded this in with poison and closed under one fault code. It
    /// is a different kind of fact: poison and closed are things a producer
    /// did, and this is memory that does not say anything coherent.
    Inconsistent {
        /// What the ring reported.
        head: u64,
        /// What the ring reported.
        tail: u64,
    },
    /// The ring moved since the batch was composed.
    Moved {
        /// What the ticket pinned.
        expected: u64,
        /// What the ring says now.
        found: u64,
    },
    /// The fire takes and the ring is empty.
    Empty,
    /// The fire puts and the ring is full.
    Full,
    /// The fire puts and the composer pinned no tail for it.
    ///
    /// A put without a pinned tail cannot be ordered against other puts to the
    /// same ring, so it is refused rather than raced.
    Unpinned,
}

impl Reason {
    /// Whether this reason will still hold however long the caller waits.
    #[must_use]
    pub fn is_permanent(self) -> bool {
        matches!(
            self,
            Reason::Poisoned | Reason::Closed | Reason::Inconsistent { .. }
        )
    }
}

/// Whether a fire may be encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// Every channel is in the state the fire needs.
    Ready,
    /// The fire is early. Waiting is the remedy.
    Retry {
        /// The channel that was not ready.
        channel: usize,
        /// What about it.
        reason: Reason,
    },
    /// The fire will never be runnable.
    Failed {
        /// The channel at fault, or `None` when the inputs themselves do not
        /// line up.
        channel: Option<usize>,
        /// What about it.
        reason: Reason,
    },
    /// The three per-channel tables are not the same length.
    ///
    /// Its own variant rather than a [`Readiness::Failed`] with an invented
    /// channel index: nothing here is a fact about a channel, and the C++'s
    /// single "fire/channel layout mismatch" string lost which pair disagreed.
    Mismatched {
        /// How many channels the fire has.
        channels: usize,
        /// How many effects the program declares.
        effects: usize,
        /// How many tickets the composer supplied.
        tickets: usize,
    },
}

/// One ring's four words, as a value.
///
/// The check below is a function of these four numbers and nothing else; the
/// [`ChannelState`] it was first written against was its first caller, not
/// its input. Snapshotting them makes that true in the signature — and lets
/// the device-backed ring, whose words live in a Metal buffer rather than in
/// host atomics, feed the same check instead of a hand-rolled copy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Words {
    /// The consume sequence.
    pub head: u64,
    /// The publish sequence.
    pub tail: u64,
    /// Non-zero when a producer faulted the ring.
    pub poison: u64,
    /// Non-zero when no further cell will ever arrive.
    pub closed: u64,
}

/// Check every channel a fire touches.
///
/// The three slices are parallel and must be the same length; a fire with
/// three channels has three effects and three tickets.
///
/// Channels are checked in order and the first answer wins, so the reported
/// channel is the earliest one that was not ready rather than an arbitrary one.
#[must_use]
pub fn check(channels: &[&ChannelState], effects: &[Effect], tickets: &[Ticket]) -> Readiness {
    let words: Vec<Words> = channels
        .iter()
        .map(|state| Words {
            head: state.head(),
            tail: state.tail(),
            poison: state.poison(),
            closed: state.closed(),
        })
        .collect();
    check_words(&words, effects, tickets)
}

/// [`check`], against ring words already in hand.
///
/// This is the whole check; [`check`] is the snapshot of a [`ChannelState`]
/// fed into it. The device ring takes this entry, because its words live in
/// shared GPU memory and are read once, atomically, rather than four times.
#[must_use]
pub fn check_words(channels: &[Words], effects: &[Effect], tickets: &[Ticket]) -> Readiness {
    if channels.len() != effects.len() || channels.len() != tickets.len() {
        return Readiness::Mismatched {
            channels: channels.len(),
            effects: effects.len(),
            tickets: tickets.len(),
        };
    }

    for (index, ((state, effect), ticket)) in channels.iter().zip(effects).zip(tickets).enumerate()
    {
        let head = state.head;
        let tail = state.tail;
        let failed = |reason| Readiness::Failed {
            channel: Some(index),
            reason,
        };
        let retry = |reason| Readiness::Retry {
            channel: index,
            reason,
        };

        if state.poison != 0 {
            return failed(Reason::Poisoned);
        }
        if state.closed != 0 {
            return failed(Reason::Closed);
        }
        if tail < head {
            return failed(Reason::Inconsistent { head, tail });
        }
        let live = tail - head;

        if ticket.expected_head != NO_TICKET && head != ticket.expected_head {
            return retry(Reason::Moved {
                expected: ticket.expected_head,
                found: head,
            });
        }
        if effect.requires_full && live == 0 {
            return retry(Reason::Empty);
        }
        if effect.requires_empty && live >= u64::from(effect.capacity) {
            return retry(Reason::Full);
        }

        if effect.put {
            if ticket.expected_tail == NO_TICKET {
                return retry(Reason::Unpinned);
            }
            if tail != ticket.expected_tail {
                return retry(Reason::Moved {
                    expected: ticket.expected_tail,
                    found: tail,
                });
            }
            // A fire that takes from this ring first frees a slot, so it may
            // put against a ring that is exactly full.
            let credit = u64::from(effect.take && effect.requires_full);
            if live >= u64::from(effect.capacity) + credit {
                return retry(Reason::Full);
            }
        }
    }
    Readiness::Ready
}

#[cfg(test)]
mod tests {
    use super::super::value::Value;
    use super::*;
    use tensor_ir::DType;

    /// A ring with `filled` cells already in it.
    fn ring(capacity: usize, filled: usize) -> ChannelState {
        let state = ChannelState::host(DType::F32, 1, capacity);
        for _ in 0..filled {
            assert!(
                state.push(&Value::F32(vec![0.0])),
                "the ring must have room for its own capacity"
            );
        }
        state
    }

    fn takes(capacity: u32) -> Effect {
        Effect {
            requires_full: true,
            take: true,
            capacity,
            ..Effect::default()
        }
    }

    fn puts(capacity: u32) -> Effect {
        Effect {
            requires_empty: true,
            put: true,
            capacity,
            ..Effect::default()
        }
    }

    fn pinned(state: &ChannelState) -> Ticket {
        Ticket {
            expected_head: state.head(),
            expected_tail: state.tail(),
        }
    }

    #[test]
    fn a_fire_whose_channels_are_all_in_the_state_it_needs_is_ready() {
        let state = ring(4, 1);
        assert_eq!(
            check(&[&state], &[takes(4)], &[pinned(&state)]),
            Readiness::Ready
        );
    }

    /// Early is not broken. A consumer ahead of its producer is the normal
    /// case in a pipelined batch, and reporting it as a failure would kill a
    /// request that had done nothing wrong.
    #[test]
    fn a_take_from_an_empty_ring_is_a_retry_not_a_failure() {
        let state = ring(4, 0);
        let outcome = check(&[&state], &[takes(4)], &[pinned(&state)]);
        assert_eq!(
            outcome,
            Readiness::Retry {
                channel: 0,
                reason: Reason::Empty
            }
        );
        assert!(!matches!(outcome, Readiness::Failed { .. }));
    }

    #[test]
    fn a_put_to_a_full_ring_is_a_retry() {
        let state = ring(2, 2);
        assert_eq!(
            check(&[&state], &[puts(2)], &[pinned(&state)]),
            Readiness::Retry {
                channel: 0,
                reason: Reason::Full
            }
        );
    }

    /// The credit is what keeps a stage that consumes and produces on one ring
    /// from deadlocking the moment the ring reaches capacity: the take it
    /// performs first is the room the put needs.
    #[test]
    fn a_fire_that_takes_and_puts_on_one_ring_may_put_at_exactly_capacity() {
        let state = ring(2, 2);
        let effect = Effect {
            requires_full: true,
            take: true,
            put: true,
            capacity: 2,
            ..Effect::default()
        };
        assert_eq!(
            check(&[&state], &[effect], &[pinned(&state)]),
            Readiness::Ready,
            "the take frees the slot the put needs; without the credit this \
             stage can never run again once its ring fills"
        );
    }

    /// And the credit is exactly one: a fire that only puts gets none.
    #[test]
    fn a_fire_that_only_puts_gets_no_credit() {
        let state = ring(2, 2);
        assert!(matches!(
            check(&[&state], &[puts(2)], &[pinned(&state)]),
            Readiness::Retry {
                reason: Reason::Full,
                ..
            }
        ));
    }

    /// Someone else took the cell this fire was composed against. Running it
    /// would consume a different one.
    #[test]
    fn a_ring_that_moved_since_composition_is_a_retry_naming_both_sequences() {
        let state = ring(4, 2);
        let ticket = Ticket {
            expected_head: state.head() + 1,
            expected_tail: state.tail(),
        };
        assert_eq!(
            check(&[&state], &[takes(4)], &[ticket]),
            Readiness::Retry {
                channel: 0,
                reason: Reason::Moved {
                    expected: 1,
                    found: 0
                }
            }
        );
    }

    /// An unpinned end is not a condition, so it is not checked.
    #[test]
    fn an_unpinned_head_skips_the_ticket_check() {
        let state = ring(4, 2);
        assert_eq!(
            check(&[&state], &[takes(4)], &[Ticket::default()]),
            Readiness::Ready
        );
    }

    /// But a put with an unpinned tail cannot be ordered against other puts to
    /// the same ring, so it waits rather than racing.
    #[test]
    fn a_put_with_no_pinned_tail_is_refused() {
        let state = ring(4, 0);
        assert_eq!(
            check(&[&state], &[puts(4)], &[Ticket::default()]),
            Readiness::Retry {
                channel: 0,
                reason: Reason::Unpinned
            }
        );
    }

    /// Closed is permanent: waiting cannot produce a cell that will never be
    /// written.
    #[test]
    fn a_closed_channel_fails_permanently() {
        let state = ring(4, 0);
        state.close();
        let outcome = check(&[&state], &[takes(4)], &[Ticket::default()]);
        let Readiness::Failed { channel, reason } = outcome else {
            panic!("a closed channel is not something waiting fixes: {outcome:?}");
        };
        assert_eq!(channel, Some(0));
        assert_eq!(reason, Reason::Closed);
        assert!(reason.is_permanent());
    }

    /// The three parallel tables come from three different places, so nothing
    /// structural makes them agree.
    #[test]
    fn tables_of_different_lengths_are_their_own_answer() {
        let state = ring(4, 1);
        assert_eq!(
            check(&[&state], &[takes(4), takes(4)], &[Ticket::default()]),
            Readiness::Mismatched {
                channels: 1,
                effects: 2,
                tickets: 1
            }
        );
    }

    /// The earliest unready channel is the one named, so the answer does not
    /// depend on iteration order the caller cannot see.
    #[test]
    fn the_first_unready_channel_is_the_one_reported() {
        let full = ring(4, 1);
        let empty = ring(4, 0);
        assert_eq!(
            check(
                &[&full, &empty, &empty],
                &[takes(4); 3],
                &[Ticket::default(); 3]
            ),
            Readiness::Retry {
                channel: 1,
                reason: Reason::Empty
            }
        );
    }

    /// Retry reasons are not permanent; that distinction is the whole point of
    /// keeping three outcomes instead of two.
    #[test]
    fn only_producer_faults_and_incoherent_rings_are_permanent() {
        for reason in [Reason::Poisoned, Reason::Closed] {
            assert!(reason.is_permanent(), "{reason:?}");
        }
        assert!(Reason::Inconsistent { head: 2, tail: 1 }.is_permanent());
        for reason in [
            Reason::Empty,
            Reason::Full,
            Reason::Unpinned,
            Reason::Moved {
                expected: 1,
                found: 0,
            },
        ] {
            assert!(!reason.is_permanent(), "{reason:?}");
        }
    }
}
