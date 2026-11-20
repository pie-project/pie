use std::sync::atomic::AtomicU64;

#[derive(Debug, Default)]
pub struct PlannerStats {
    /// Plan passes run (`ResidencyPlanner::plan`).
    pub plan_passes: AtomicU64,
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
