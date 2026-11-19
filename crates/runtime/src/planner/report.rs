use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::store::rs::RsUnits;

use super::grant::Demand;
use super::proc::Residency;
use super::wait::WaitKind;
use super::{ProcessId, ResidencyPlanner};

impl ResidencyPlanner {
    pub(super) fn report_evicted(self: &Arc<Self>, pid: ProcessId, freed: u32, rs_freed: RsUnits) {
        self.with_inner(|inner| {
            // One that left meanwhile is settled by its departure.
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            inner.evicting.remove(&pid);
            debug_assert_eq!(proc.state, Residency::Evicting);
            proc.state = Residency::Evicted;
            // Parked again: the next suspended bind may come back.
            if inner.bind_restoring == Some(pid) {
                inner.bind_restoring = None;
            }
            // What waits for its move to end ([`ResidencyPlanner::wait_not_moving`]).
            proc.signal.notify_waiters();
            let demand = Demand {
                kv_pages: freed,
                rs: rs_freed,
            };
            // A holder owes nothing: whoever takes its set asks for it back.
            if proc.idle || proc.held || proc.holder {
                proc.parked = Some(demand);
                return;
            }
            inner.enqueue(pid, WaitKind::Restore { demand });
        });
        crate::policy::report(crate::policy::Event::Paused(pid));
        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(target: "pie::kv", kv_pages = freed, "paused");
        self.vacate_held_seats();
        self.re_arm_idle_reclaim();
        self.poke();
    }

    pub(super) fn eviction_failed(self: &Arc<Self>, pid: ProcessId) {
        self.eviction_failed_inner(pid, false, false);
    }

    /// An eviction whose copy to the host failed: `pid`'s state could not
    /// be moved, and a second try moves it no more than the first, so it is
    /// never tried again. The process is failed by name, the copy's reason
    /// given; it stands among the kills in flight, which are no victims and
    /// free its pages as it ends, and the planner looks for the next one.
    pub(super) fn eviction_copy_failed(self: &Arc<Self>, pid: ProcessId, error: &str) {
        let signal = self.with_inner(|inner| {
            // One that left meanwhile is settled by its departure.
            if !inner.procs.contains_key(&pid) {
                return None;
            }
            inner.settle_eviction(pid);
            inner.killing.insert(pid);
            let proc = inner.procs.get_mut(&pid)?;
            if proc.state == Residency::Evicting {
                proc.state = Residency::Resident;
                // Never parked: its bind goes again as it stands.
                proc.bind_room = None;
            }
            Some(proc.signal.clone())
        });
        self.stats.eviction_faults.fetch_add(1, Ordering::Relaxed);
        crate::scheduler::worker::notify_process_resume(pid);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        let reason = format!(
            "planner: this process's state could not be moved off the device to make room, \
             so it is failed rather than tried again: {error}"
        );
        tracing::error!(pid = %pid, %reason, "planner: failing a process whose eviction could not land");
        crate::inferlet::process::terminate_by(pid, Err(reason), crate::policy::By::Host);
        self.poke();
    }

    pub(super) fn eviction_found_nothing(self: &Arc<Self>, pid: ProcessId) {
        self.eviction_failed_inner(pid, false, true);
    }

    pub(super) fn eviction_failed_host_swap_full(self: &Arc<Self>, pid: ProcessId) {
        self.record_host_swap_exhaustion();
        self.eviction_failed_inner(pid, true, false);
    }

    fn eviction_failed_inner(
        self: &Arc<Self>,
        pid: ProcessId,
        host_swap_full: bool,
        found_nothing: bool,
    ) {
        let signal = self.with_inner(|inner| {
            // One that left meanwhile is settled by its departure.
            if inner.procs.contains_key(&pid) {
                inner.settle_eviction(pid);
            }
            if host_swap_full {
                inner.host_swap_blocked.insert(pid);
            }
            if found_nothing {
                inner.prepare_blocked.insert(pid);
            }
            let proc = inner.procs.get_mut(&pid)?;
            if proc.state == Residency::Evicting {
                proc.state = Residency::Resident;
                // Never parked: its bind goes again as it stands.
                proc.bind_room = None;
            }
            Some(proc.signal.clone())
        });
        self.stats
            .eviction_rollbacks
            .fetch_add(1, Ordering::Relaxed);
        crate::scheduler::worker::notify_process_resume(pid);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
    }

    pub(super) fn clear_host_swap_blocks(&self) {
        let cleared = self.with_inner(|inner| {
            if inner.host_swap_blocked.is_empty() && inner.prepare_blocked.is_empty() {
                return false;
            }
            inner.host_swap_blocked.clear();
            inner.prepare_blocked.clear();
            true
        });
        if cleared {
            self.stats
                .host_swap_unblocks
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn report_restored(self: &Arc<Self>, pid: ProcessId) {
        self.clear_host_swap_blocks();
        let signal = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            debug_assert_eq!(proc.state, Residency::Restoring);
            proc.state = Residency::Resident;
            proc.progressed = false;
            proc.roles_short = None;
            proc.roles_free_retry = None;
            proc.roles_free_short = None;
            proc.roles_short_gives = None;
            proc.roles_short_mismatch = None;
            // Back with room for it: the bind goes again.
            proc.bind_room = None;
            Some(proc.signal.clone())
        });
        // Resident first, then every set it holds now unfenced: one
        // registered while it was away was born fenced
        // ([`Self::register_kv_working_set`]) and is read here, and one
        // registered from now on is born unfenced.
        let (model, engine) = self.port.locus();
        for handle in crate::inferlet::process::residency::kv_suspend_handles(pid, model, engine) {
            handle.unfence();
        }
        crate::scheduler::worker::notify_process_resume(pid);
        crate::policy::report(crate::policy::Event::Resumed(pid));
        self.stats.restores.fetch_add(1, Ordering::Relaxed);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
    }

    /// A restore that failed at a step of its own (its kv, its state or its
    /// role slots prepared, or its copy back): a second try fails as the
    /// first did, so it is never queued again. The process is failed by
    /// name, `reason` given, and counted as a restore fault.
    pub(super) fn restore_faulted(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        self.stats.restore_faults.fetch_add(1, Ordering::Relaxed);
        self.restore_failed(pid, reason);
    }

    pub(super) fn restore_failed(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        let live = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return false;
            };
            if proc.state != Residency::Restoring {
                return false;
            }
            proc.state = Residency::Evicted;
            true
        });
        self.stats.restore_failures.fetch_add(1, Ordering::Relaxed);
        if live {
            self.fail_restore_loud(pid, reason);
        } else {
            self.poke();
        }
    }

    /// The guest-state room the bind that suspended `pid` asks, while
    /// that bind is pending ([`super::proc::Proc::bind_room`]).
    pub(super) fn pending_bind(&self, pid: ProcessId) -> Option<super::proc::PendingBind> {
        self.with_inner(|inner| inner.procs.get(&pid)?.bind_room)
    }

    /// `pid`'s restore found no device room for its parked role slots, which
    /// a budget holds: it is queued again, a waiter like any admission, and
    /// served once a completion frees room or an eviction makes it
    /// ([`super::proc::Proc::roles_short`]). No failure.
    /// `need` is the runs of whole chunks its unpark is short.
    /// Why `pid`'s last unpark came back short, kept for a stall census.
    pub(super) fn note_roles_short_why(&self, pid: ProcessId, why: String) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.roles_short_why = Some(why);
            }
        });
    }

    /// `refused_free_runs`: the free runs of the need's length the refusal
    /// itself saw, where it says ([`crate::store::roles::RolesError::unpark_free_runs`]).
    /// A refusal that saw runs of its need standing and was short all the
    /// same understates what the unpark takes; one that saw too few is a
    /// plain short head, sent again as soon as the runs stand.
    pub(super) fn requeue_restore_roles(
        self: &Arc<Self>,
        pid: ProcessId,
        need: (u32, u32),
        refused_free_runs: Option<u32>,
    ) {
        let pages = self.port.chunks_as_pages(need.0.saturating_mul(need.1));
        // The map as the refusal leaves it: its epoch and its runs of the
        // need's length.
        let gives = self.port.room_gives();
        let map_runs = self.port.guest_free_runs(need.0);
        self.stats
            .restore_roles_short
            .fetch_add(1, Ordering::Relaxed);
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.roles_short = Some(super::proc::PendingBind {
                    run: need.0,
                    pieces: need.1,
                    pages,
                });
                // Refused while runs of this need stood (its refusal's own
                // view): the need understates its unpark. Refused with too
                // few standing, it is a plain short head, whatever it was
                // sent on.
                proc.roles_free_retry = None;
                proc.roles_short_gives = Some(gives);
                proc.roles_short_mismatch = None;
                let refused_saw = refused_free_runs.is_some_and(|free| free >= need.1);
                let map_shows = map_runs >= need.1;
                if refused_saw || map_shows {
                    // Short while runs of its need stood, in its own view or
                    // the map's: the need understates its unpark.
                    proc.roles_free_short = Some(need);
                    proc.roles_free_short_at = map_runs.max(refused_free_runs.unwrap_or(0));
                    if map_shows && !refused_saw {
                        proc.roles_short_mismatch = Some((map_runs, refused_free_runs));
                    }
                } else {
                    proc.roles_free_short = None;
                }
            }
        });
        self.requeue_restore(pid);
    }

    /// Role slots out of service for good stand at `slots` now (an aborted
    /// move's copies never settled): the status says so, and the plan looks
    /// again, as the move's items no longer move.
    pub(crate) fn note_leaked(self: &Arc<Self>, slots: u64) {
        self.stats.leaked_role_slots.store(slots, Ordering::Relaxed);
        self.poke();
    }

    /// More of `pid`'s pages were swapped out since its restore was quoted:
    /// it is queued again, to be quoted anew when served. No failure.
    pub(super) fn requeue_restore(self: &Arc<Self>, pid: ProcessId) {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            if proc.state != Residency::Restoring {
                return;
            }
            proc.state = Residency::Evicted;
            if inner.bind_restoring == Some(pid) {
                inner.bind_restoring = None;
            }
            inner.enqueue(
                pid,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
        });
        self.poke();
    }

    fn fail_restore_loud(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        let reason = format!("KV restore failed: {reason}");
        // The engine's scheduler gone: the worker is shutting down, and the
        // restore was cut by it, no failure of the process's (logs after
        // SIGTERM: "engine 0 has no scheduler").
        let (_, engine) = self.port.locus();
        if crate::scheduler::scheduler_handle(engine).is_err() {
            tracing::debug!(pid = %pid, %reason, "planner: a restore cut by shutdown");
        } else {
            tracing::error!(pid = %pid, %reason, "planner: failing evicted process loud");
        }
        crate::inferlet::process::terminate_by(pid, Err(reason), crate::policy::By::Host);
    }

    pub(super) fn record_host_swap_exhaustion(&self) {
        self.stats
            .host_swap_exhaustions
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one engine copy of a residency move and, for kv, its time.
    pub(crate) fn record_tier_copy(&self, copy: &crate::scheduler::TierMove, elapsed: Duration) {
        use crate::scheduler::TierKind::{Kv, KvSlots, RoleSlots, RsSlots};
        use ::engine::transfer::MemoryDomain::LocalDisk;
        let stats = &self.stats;
        let (count, us) = match (copy.kind, copy.tier, copy.out) {
            (Kv | KvSlots(_), LocalDisk, true) => {
                (&stats.disk_write_pages, Some(&stats.disk_write_us))
            }
            (Kv | KvSlots(_), LocalDisk, false) => {
                (&stats.disk_read_pages, Some(&stats.disk_read_us))
            }
            (Kv | KvSlots(_), _, true) => (&stats.d2h_pages, Some(&stats.d2h_copy_us)),
            (Kv | KvSlots(_), _, false) => (&stats.h2d_pages, Some(&stats.h2d_copy_us)),
            (RsSlots(_) | RoleSlots(_), _, true) => (&stats.d2h_rs_slots, None),
            (RsSlots(_) | RoleSlots(_), _, false) => (&stats.h2d_rs_slots, None),
        };
        count.fetch_add(copy.device.len() as u64, Ordering::Relaxed);
        if copy.source.is_some() {
            stats
                .host_disk_pages
                .fetch_add(copy.device.len() as u64, Ordering::Relaxed);
        }
        if let Some(us) = us {
            let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
            us.fetch_add(micros, Ordering::Relaxed);
        }
    }
}
