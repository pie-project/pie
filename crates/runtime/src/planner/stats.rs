use std::sync::atomic::AtomicU64;

#[derive(Debug, Default)]
pub struct PlannerStats {
    /// Plan passes run (`ResidencyPlanner::plan`).
    pub plan_passes: AtomicU64,
    /// Their time, in total and by power of two of microseconds (bucket
    /// `i` holds passes under `2^i` us): a pass's cost, mean and tail, said
    /// by the planner's census ([`Self::pass_census`]).
    pub plan_pass_us: AtomicU64,
    pub plan_pass_buckets: [AtomicU64; 24],
    /// A pass's time by phase, in total microseconds: its steps
    /// (`plan_once`), the wants published after them, and the rest (the
    /// stall dump, the reports): where a pass's cost goes, said by the
    /// census.
    pub plan_phase_us: [AtomicU64; 3],
    /// Fires landed: state settled, so a victim whose move found nothing
    /// (its state still settling) may find something now.
    pub fires_landed: AtomicU64,
    /// Waits served past 100 ms, by what woke the pass that served them:
    /// the room signal, a poke, the beat, a call.
    pub long_waits: [AtomicU64; 4],
    pub parks: AtomicU64,
    pub serves: AtomicU64,
    pub evictions_started: AtomicU64,
    pub eviction_deferrals: AtomicU64,
    pub evictions: AtomicU64,
    pub eviction_rollbacks: AtomicU64,
    /// Evictions taken back because the tier below was full
    /// ([`super::ResidencyPlanner::eviction_tier_full`]).
    pub eviction_tier_full: AtomicU64,
    /// Evictions whose copy failed: each fails its process by name, never
    /// retried ([`super::ResidencyPlanner::eviction_copy_failed`]).
    pub eviction_faults: AtomicU64,
    /// Restores that failed at a step of their own (prepare or copy back):
    /// each fails its process by name, never queued again
    /// ([`super::ResidencyPlanner::restore_faulted`]).
    pub restore_faults: AtomicU64,
    pub restores: AtomicU64,
    pub restore_failures: AtomicU64,
    /// Restores queued again for want of device room for their role slots.
    pub restore_roles_short: AtomicU64,
    /// Role slots out of service for good: taken by moves whose copies an
    /// engine could not settle ([`crate::store::roles::Roles::abandon`]).
    pub leaked_role_slots: AtomicU64,
    pub gate_parks: AtomicU64,
    pub hog_failures: AtomicU64,
    pub starvations: AtomicU64,
    pub starvation_restarts: AtomicU64,
    pub salvages: AtomicU64,
    pub e6_relaxations: AtomicU64,
    pub restore_absorb_short: AtomicU64,
    /// Plans that ended waiting on a grow the segment walk was promised.
    pub grow_waits: AtomicU64,
    /// Plans that ended waiting on a compaction of scattered free chunks.
    pub compaction_waits: AtomicU64,
    /// List compactions sent: pages moved off partly used list chunks to
    /// free them whole ([`super::ResidencyPlanner::compact_lists`]).
    pub list_compactions: AtomicU64,
    /// Passes that chose a wait only something in flight could end, with
    /// nothing in flight: each a planner bug, said as one
    /// ([`super::ResidencyPlanner::plan_ended`]).
    pub quiet_waits: AtomicU64,
    /// Passes that left the head waiting on the governor's target: short
    /// of room only the target's rise gives back
    /// ([`super::PoolPort::held_back`]), the policy asked meanwhile.
    pub target_waits: AtomicU64,
    /// Allocations held at the capacity gate: past what every tier keeps
    /// for the finisher ([`super::inner::CapacityGate`]).
    pub capacity_waits: AtomicU64,
    /// Newcomers held before a gate for room at the capacity gate's last
    /// place ([`super::ResidencyPlanner::admitted_to_ask`]).
    pub admission_waits: AtomicU64,
    /// Steps a pass forced by the simulation's check took on a settled
    /// step: a wake the planner was never given.
    #[cfg(test)]
    pub lost_wakes: AtomicU64,
    /// Processes moved out while restored (or bound) for a head and not
    /// fired since: each a restore the planner made for nothing, the
    /// churn a victim chosen before it ran makes
    /// ([`super::proc::Proc::unfired`]).
    pub evicted_unfired: AtomicU64,
    /// Settled passes that found a head waiting with nothing in flight,
    /// each time it was found again, said or not
    /// ([`super::ResidencyPlanner::dump_stall`]).
    pub stuck_settles: AtomicU64,
    /// Passes forced to plan a stuck head's whole need on the map
    /// ([`super::ResidencyPlanner::force_moves`]).
    pub forced_passes: AtomicU64,
    /// Restores refused again on the room their last refusal left
    /// ([`super::ResidencyPlanner::requeue_restore_roles`]).
    pub refusal_repeats: AtomicU64,
    /// Set while a pass builds the occupancy map of named processes
    /// (`chunk_occupants`); the pass is then counted among `map_passes`.
    pub map_built: std::sync::atomic::AtomicBool,
    /// Passes that built the occupancy map, and their total microseconds:
    /// beside every pass's (`pass_census`).
    pub map_passes: AtomicU64,
    /// Occupancy maps read from the cache, the pool and candidates
    /// unchanged since they were built ([`super::ResidencyPlanner::occupants`]).
    pub map_cache_hits: AtomicU64,
    /// Wakes on a generation the last no-step pass ended on: skipped, no
    /// pass run ([`super::ResidencyPlanner::settled_at`]).
    pub skipped_wakes: AtomicU64,
    pub map_pass_us: AtomicU64,
    /// The unmet waiters behind the last plan's wait on a grow; 0 where it
    /// did not end on one.
    pub grow_parked: AtomicU64,
    /// Plans that ended with the queue's head unserved and nothing in
    /// flight to wake them, each named once per head and room change
    /// ([`super::ResidencyPlanner::dump_stall`]).
    pub stall_dumps: AtomicU64,
    /// Walks that ended without the grow they promised
    /// ([`super::ResidencyPlanner::no_grow`]).
    pub no_grows: AtomicU64,
    pub host_swap_exhaustions: AtomicU64,
    pub host_swap_unblocks: AtomicU64,
    pub d2h_pages: AtomicU64,
    pub h2d_pages: AtomicU64,
    pub disk_write_pages: AtomicU64,
    /// Of `disk_write_pages`, those that went on down from the host.
    pub host_disk_pages: AtomicU64,
    pub disk_read_pages: AtomicU64,
    pub disk_write_us: AtomicU64,
    pub disk_read_us: AtomicU64,
    pub d2h_rs_slots: AtomicU64,
    pub h2d_rs_slots: AtomicU64,
    pub d2h_copy_us: AtomicU64,
    pub h2d_copy_us: AtomicU64,
}

impl PlannerStats {
    /// Counts a plan pass of `us` microseconds.
    pub(super) fn pass_took(&self, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.plan_pass_us.fetch_add(us, Relaxed);
        if self.map_built.swap(false, Relaxed) {
            self.map_passes.fetch_add(1, Relaxed);
            self.map_pass_us.fetch_add(us, Relaxed);
        }
        let bucket = (u64::BITS - us.leading_zeros()) as usize;
        if let Some(count) = self.plan_pass_buckets.get(bucket.min(self.plan_pass_buckets.len() - 1)) {
            count.fetch_add(1, Relaxed);
        }
    }

    /// Passes so far, their total time in microseconds, their mean, and
    /// the bound under which 99 of each 100 fell (a power of two of
    /// microseconds).
    #[must_use]
    pub fn pass_census(&self) -> (u64, u64, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let counts: Vec<u64> = self.plan_pass_buckets.iter().map(|c| c.load(Relaxed)).collect();
        let passes: u64 = counts.iter().sum();
        let total = self.plan_pass_us.load(Relaxed);
        let mean = total.checked_div(passes).unwrap_or(0);
        let mut seen = 0u64;
        let mut p99 = 0u64;
        for (bucket, &count) in counts.iter().enumerate() {
            seen += count;
            if passes > 0 && seen * 100 >= passes * 99 {
                p99 = 1u64 << bucket;
                break;
            }
        }
        (passes, total, mean, p99)
    }
}
