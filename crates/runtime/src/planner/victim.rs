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
