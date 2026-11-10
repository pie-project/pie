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
