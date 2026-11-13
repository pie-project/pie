use super::ProcessId;
use super::rank::Rank;
use super::reclaim::Reclaim;
use super::wait::EntryKey;

#[derive(Clone, Copy)]
pub(super) struct Victim {
    pub(super) pid: ProcessId,
    pub(super) rank: Rank,
    pub(super) e6_fresh: bool,
}

pub(super) struct VictimSet {
    pub(super) head: EntryKey,
    pub(super) head_pid: ProcessId,
    pub(super) deficit: Reclaim,
    /// Of `deficit.pages`, those that stand for state the head lacks: kv
    /// that parks frees chunks any shape opens ([`super::PoolPort::rs_as_pages`]).
    pub(super) rs_pages: u32,
    /// On a shared pool, the state the head lacks that `rs_pages` stands
    /// for: what a victim must park on the host to free its chunks.
    pub(super) rs_short: crate::store::rs::RsUnits,
    /// The head is a restore: it may take the host's restore reserve.
    pub(super) restore: bool,
    pub(super) members: Vec<Victim>,
}

impl VictimSet {
    pub(super) fn preferred(&self) -> Vec<(ProcessId, Rank)> {
        self.members
            .iter()
            .filter(|v| v.e6_fresh)
            .map(|v| (v.pid, v.rank))
            .collect()
    }
}
