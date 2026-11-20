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
    /// that parks frees chunks any shape opens ([`super::PoolPort::rs_as_pages`]),
    /// and the runs of chunks a bind short of guest-state room asks
    /// ([`super::PoolPort::chunks_as_pages`]).
    pub(super) rs_pages: u32,
    /// Of `deficit.pages`, the kv the head itself still lacks past what
    /// is free and on its way: its own kv need, never the queue's quantum
    /// (other waiters' one-page asks), and none for a bind. Only this waits
    /// on room below for kv ([`super::ResidencyPlanner::plan_eviction`]).
    pub(super) own_pages: u32,
    /// On a shared pool, the state the head lacks that `rs_pages` stands
    /// for: what a victim must park on the host to free its chunks.
    pub(super) rs_short: crate::store::rs::RsUnits,
    /// The head is a restore: it may take the host's restore reserve.
    pub(super) restore: bool,
    pub(super) members: Vec<Victim>,
}

impl VictimSet {
    /// The candidates an eviction pass reads: the fresh ([`Self::preferred`]),
    /// or with nothing in flight (`quiet`) every member, nothing coming to
    /// move the rest on.
    pub(super) fn candidates(&self, quiet: bool) -> Vec<(ProcessId, Rank)> {
        if quiet {
            self.members.iter().map(|v| (v.pid, v.rank)).collect()
        } else {
            self.preferred()
        }
    }

    pub(super) fn preferred(&self) -> Vec<(ProcessId, Rank)> {
        self.members
            .iter()
            .filter(|v| v.e6_fresh)
            .map(|v| (v.pid, v.rank))
            .collect()
    }
}

/// Where an eviction plan stopped ([`super::ResidencyPlanner::plan_eviction`]),
/// said by the stall dump: a head left unserved with nothing in flight
/// names the check that dropped its plan.
#[derive(Clone, Debug)]
pub(super) struct PlanEnd {
    pub(super) head: EntryKey,
    pub(super) ended: &'static str,
    /// The room below (kv swap, or host and disk rows for state) could
    /// not take what the move parks.
    pub(super) host_short: bool,
    /// Candidates whose move the room below could not take.
    pub(super) unhostable: Vec<ProcessId>,
}
