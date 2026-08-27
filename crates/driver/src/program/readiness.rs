use super::channel::ChannelState;

pub const NO_TICKET: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Effect {
    pub requires_full: bool,

    pub requires_empty: bool,

    pub take: bool,

    pub put: bool,

    pub capacity: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket {
    pub expected_head: u64,

    pub expected_tail: u64,
}

impl Default for Ticket {
    fn default() -> Self {
        Ticket {
            expected_head: NO_TICKET,
            expected_tail: NO_TICKET,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Poisoned,

    Closed,

    Inconsistent { head: u64, tail: u64 },

    Moved { expected: u64, found: u64 },

    Empty,

    Full,

    Unpinned,
}

impl Reason {
    #[must_use]
    pub fn is_permanent(self) -> bool {
        matches!(
            self,
            Reason::Poisoned | Reason::Closed | Reason::Inconsistent { .. }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    Ready,

    Retry {
        channel: usize,

        reason: Reason,
    },

    Failed {
        channel: Option<usize>,

        reason: Reason,
    },

    Mismatched {
        channels: usize,

        effects: usize,

        tickets: usize,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Words {
    pub head: u64,

    pub tail: u64,

    pub poison: u64,

    pub closed: u64,
}

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

            let credit = u64::from(effect.take && effect.requires_full);
            if live >= u64::from(effect.capacity) + credit {
                return retry(Reason::Full);
            }
        }
    }
    Readiness::Ready
}
