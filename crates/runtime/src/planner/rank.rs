/// The policy key first, larger earlier, then a holder's retention key, the
/// same way, then age, older earlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Rank {
    key: std::cmp::Reverse<i64>,
    kept: std::cmp::Reverse<i64>,
    pub(super) seq: u64,
}

impl Rank {
    pub(super) fn of(key: i64, seq: u64) -> Self {
        Self {
            key: std::cmp::Reverse(key),
            kept: std::cmp::Reverse(0),
            seq,
        }
    }

    pub(super) fn kept(self, kept: i64) -> Self {
        Self {
            kept: std::cmp::Reverse(kept),
            ..self
        }
    }
}

/// The order processes give up the device in, first first: the one
/// function every park and eviction picks its victims by. The policy's key
/// leads (lower earlier), then a holder's retention key (lower earlier);
/// past what the policy says, the process made resident last (admitted or
/// restored) goes first, so those running longest keep their progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct VictimOrder {
    key: i64,
    kept: i64,
    came: std::cmp::Reverse<u64>,
}

impl VictimOrder {
    pub(super) fn of(key: i64, kept: i64, came: u64) -> Self {
        Self {
            key,
            kept,
            came: std::cmp::Reverse(came),
        }
    }
}
