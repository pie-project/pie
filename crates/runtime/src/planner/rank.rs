/// The policy key first, larger earlier, then age, older earlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Rank {
    key: std::cmp::Reverse<i64>,
    pub(super) seq: u64,
}

impl Rank {
    pub(super) fn of(key: i64, seq: u64) -> Self {
        Self {
            key: std::cmp::Reverse(key),
            seq,
        }
    }
}
