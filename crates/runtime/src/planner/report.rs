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
            inner.settle_eviction(pid);
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            debug_assert_eq!(proc.state, Residency::Evicting);
            proc.state = Residency::Evicted;
            let demand = Demand {
                kv_pages: freed,
                rs: rs_freed,
            };
            if proc.idle || proc.held {
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
            inner.settle_eviction(pid);
            if host_swap_full {
                inner.host_swap_blocked.insert(pid);
            }
            if found_nothing {
                inner.prepare_blocked.insert(pid);
            }
            let proc = inner.procs.get_mut(&pid)?;
            if proc.state == Residency::Evicting {
                proc.state = Residency::Resident;
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
        let (model, engine) = self.port.locus();
        for handle in crate::inferlet::process::residency::kv_suspend_handles(pid, model, engine) {
            handle.unfence();
        }
        let signal = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            debug_assert_eq!(proc.state, Residency::Restoring);
            proc.state = Residency::Resident;
            proc.restore_retried = false;
            proc.progressed = false;
            Some(proc.signal.clone())
        });
        crate::scheduler::worker::notify_process_resume(pid);
        crate::policy::report(crate::policy::Event::Resumed(pid));
        self.stats.restores.fetch_add(1, Ordering::Relaxed);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        self.poke();
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

    /// A `failure` spends its one retry; none (pages it shares moved since its
    /// grant) is no failure.
    pub(super) fn requeue_restore(self: &Arc<Self>, pid: ProcessId, failure: Option<&str>) {
        let spent = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return false;
            };
            if proc.state != Residency::Restoring {
                return false;
            }
            proc.state = Residency::Evicted;
            if failure.is_some() {
                if proc.restore_retried {
                    return true;
                }
                proc.restore_retried = true;
            }
            inner.enqueue(
                pid,
                WaitKind::Restore {
                    demand: Demand::default(),
                },
            );
            false
        });
        if failure.is_some() {
            self.stats.restore_failures.fetch_add(1, Ordering::Relaxed);
        }
        match failure {
            Some(reason) if spent => self.fail_restore_loud(pid, reason),
            _ => self.poke(),
        }
    }

    fn fail_restore_loud(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        let reason = format!("KV restore failed: {reason}");
        tracing::error!(pid = %pid, %reason, "planner: failing evicted process loud");
        crate::inferlet::process::terminate_by(pid, Err(reason), crate::policy::By::Host);
    }

    pub(super) fn record_host_swap_exhaustion(&self) {
        self.stats
            .host_swap_exhaustions
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one engine copy of a residency move and, for kv, its time.
    pub(crate) fn record_tier_copy(&self, copy: &crate::scheduler::TierMove, elapsed: Duration) {
        use crate::scheduler::TierKind::{Kv, Rs, RsPages, Window};
        use ::engine::transfer::MemoryDomain::LocalDisk;
        let stats = &self.stats;
        let (count, us) = match (copy.kind, copy.tier, copy.out) {
            (Kv, LocalDisk, true) => (&stats.disk_write_pages, Some(&stats.disk_write_us)),
            (Kv, LocalDisk, false) => (&stats.disk_read_pages, Some(&stats.disk_read_us)),
            (Kv, _, true) => (&stats.d2h_pages, Some(&stats.d2h_copy_us)),
            (Kv, _, false) => (&stats.h2d_pages, Some(&stats.h2d_copy_us)),
            (Rs | RsPages | Window, _, true) => (&stats.d2h_rs_slots, None),
            (Rs | RsPages | Window, _, false) => (&stats.h2d_rs_slots, None),
        };
        count.fetch_add(copy.device.len() as u64, Ordering::Relaxed);
        if let Some(us) = us {
            let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
            us.fetch_add(micros, Ordering::Relaxed);
        }
    }
}
