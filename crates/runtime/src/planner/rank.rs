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
