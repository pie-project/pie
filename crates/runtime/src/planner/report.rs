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
            #[cfg(test)]
            inner.moves.push((pid, Some(proc.evicted_for)));
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

    /// An eviction whose copy the engine refused before reading or writing
    /// any of it (a placement it will not copy: the runtime's own doing,
    /// never the process's): the move is taken back, the process stays
    /// resident, and it is no victim again until the pool moves past the
    /// refusal, said as the runtime's bug: no process is failed for it.
    /// An eviction whose copy down failed because the tier below is full
    /// (the disk tier's slot file, `StorageFull`): nothing on the device
    /// changed, so the move is taken back and the process stays resident,
    /// said by name, and it is no victim again until the pool moves past
    /// it (its block lapses, [`Self::lapse_blocks`]): moved again at once,
    /// it would find the tier as full. With the fleet stalled on it, every
    /// tier full, it fails by name instead ([`Self::fail_tiers_full`]).
    pub(super) fn eviction_tier_full(self: &Arc<Self>, pid: ProcessId, error: &str) {
        tracing::warn!(
            pid = %pid,
            %error,
            "planner: the tier below is full: the eviction is taken back"
        );
        self.stats.eviction_tier_full.fetch_add(1, Ordering::Relaxed);
        // Blocked as room below short, which a stall with nothing in flight
        // lifts ([`Self::dump_stall`]).
        self.eviction_failed_inner(pid, true, false);
        // Every tier full and the fleet stalled: the victim fails by name
        // and its room frees for the others ([`Self::fail_tiers_full`]).
        if self.is_wedged() && !self.waits_on_policy() {
            self.fail_tiers_full(Some(pid));
        }
    }

    pub(super) fn eviction_copy_refused(self: &Arc<Self>, pid: ProcessId, error: &str) {
        tracing::error!(
            pid = %pid,
            %error,
            "planner: the engine refused an eviction's copy untouched: the runtime placed state \
             it will not copy (a runtime bug); the eviction is taken back, the process kept"
        );
        self.stats.eviction_faults.fetch_add(1, Ordering::Relaxed);
        self.eviction_failed_inner(pid, false, true);
    }

    /// An eviction's copy did not land ([`::engine::Landing::Failed`]): its
    /// state is still on the device, its move taken back, and the process
    /// kept resident, a victim again at once: the copy goes again from
    /// where its state lies. Blocked until the pool moved (as a full tier
    /// is), a fleet with nothing else in flight waited on it for good.
    pub(super) fn eviction_landing_failed(self: &Arc<Self>, pid: ProcessId, error: &str) {
        tracing::warn!(
            pid = %pid,
            %error,
            "planner: an eviction's copy did not land: taken back, the process kept resident"
        );
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
        let pool_now = self.pool_now();
        let signal = self.with_inner(|inner| {
            if host_swap_full || found_nothing {
                inner.block_on(pool_now);
            }
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
                // Never parked: its bind goes again as it stands. The room
                // a bind withdrawn by this eviction waited for is kept
                // until that bind asks again ([`Self::wait_room`]): moved
                // again before it did, its restore would come back without
                // the room, short, and its process go down a second time for
                // one bind.
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

    /// `pid`'s restore waits for its instances' rebinds (`Some`, the
    /// instances), or no longer (`None`): waiting, it is nothing in flight
    /// ([`super::proc::Proc::rebind_waiting`]), and the planner plans on
    /// that at once.
    pub(super) fn rebind_waiting(self: &Arc<Self>, pid: ProcessId, waiting: Option<String>) {
        let changed = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            let changed = proc.rebind_waiting.is_some() != waiting.is_some();
            proc.rebind_waiting = waiting;
            Some(changed)
        });
        if changed == Some(true) {
            self.poke();
        }
    }

    pub(super) fn report_restored(self: &Arc<Self>, pid: ProcessId) {
        self.clear_host_swap_blocks();
        // With no bind pending, its unpark took what was held for it: the
        // rest goes back. A pending bind takes its runs from it, and its
        // take gives back the rest.
        if self.with_inner(|inner| inner.procs.get(&pid).is_none_or(|proc| proc.bind_room.is_none())) {
            self.port.release_restore(pid);
        }
        let signal = self.with_inner(|inner| {
            #[cfg(test)]
            inner.moves.push((pid, None));
            inner.arrived(pid);
            let proc = inner.procs.get_mut(&pid)?;
            debug_assert_eq!(proc.state, Residency::Restoring);
            proc.state = Residency::Resident;
            proc.rebind_waiting = None;
            proc.progressed = false;
            // Back for a waiting head (its own restore): no victim until it
            // has fired once; and what it was evicted for is past.
            proc.unfired = true;
            proc.evicted_for = None;
            proc.roles_short = None;
            proc.roles_free_retry = None;
            proc.roles_free_short = None;
            proc.roles_short_gives = None;
            proc.roles_short_runs = None;
            proc.roles_short_free = None;
            proc.last_refusal = None;
            proc.roles_short_stood = None;
            proc.roles_refused = None;
            proc.evicted_queued = false;
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
        self.port.release_restore(pid);
        let live = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return false;
            };
            if proc.state != Residency::Restoring {
                return false;
            }
            proc.state = Residency::Evicted;
            proc.rebind_waiting = None;
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
    ///
    /// The head's need becomes the refusal's own, never less than what was
    /// recorded for it: the longest run and the most pieces of the refused
    /// need, the bind that suspended it and its last refusal, kept as both
    /// (else, recorded (3,1) beside a refused (4,1), the plan funds the
    /// smaller while the restore waits on the larger). A
    /// refusal that came while runs of that need stood (in its own view or
    /// the map's) understates what the unpark takes: its need becomes one
    /// run more than stood, which the planner funds by moving the occupants
    /// of the cheapest such runs, rather than wait for the map to change on
    /// its own. Each such refusal asks one run more, so it ends: served, or
    /// past the whole pool and refused by name.
    pub(super) fn requeue_restore_roles(
        self: &Arc<Self>,
        pid: ProcessId,
        need: (u32, u32),
        refused_free_runs: Option<u32>,
    ) {
        self.port.release_restore(pid);
        let recorded = self.with_inner(|inner| {
            inner
                .procs
                .get(&pid)
                .map(|proc| proc.bind_room.into_iter().chain(proc.roles_short).collect::<Vec<_>>())
                .unwrap_or_default()
        });
        // Of what was recorded, only a need of the refusal's own length:
        // folded over a bind's run of seven, a refusal of a run of four
        // would read as runs of seven.
        let run = need.0;
        let pieces = recorded
            .iter()
            .filter(|had| had.run == run)
            .fold(need.1, |pieces, had| pieces.max(had.pieces));
        // The whole unpark, not the part its refusal named: recorded so, the
        // need is fixed as the restore takes it, and a refusal while those
        // runs stand is one the need truly understates. Its other lengths
        // are read where its need is ([`Self::head_room`]).
        let pieces = self
            .whole_unpark(pid, (run, pieces))
            .pieces_at(run)
            .min(self.port.state_chunks() / run.max(1))
            .max(pieces);
        // The runs of the length its refusal named, standing now: what it
        // goes again on. Read at its need's length (a run of seven, the
        // bind's, folded over the unpark's run of four), runs of seven
        // freed would send it again while the run of four it was refused
        // never stands.
        let refused_room = self.port.guest_free_runs(need.0);
        // The free chunks a guest's take reads (within the state budget),
        // as the refusal leaves them: what it goes again on too.
        let free_chunks = self.port.free_chunks();
        // The map as the refusal leaves it: its epoch and its runs of the
        // need's length.
        let gives = self.port.room_gives();
        let map_runs = self.port.guest_free_runs(run);
        // The refusal's own count is of runs of its own length.
        let refused_runs = refused_free_runs;
        let refused_saw = refused_runs.is_some_and(|free| free >= pieces);
        let map_shows = map_runs >= pieces;
        let stood = map_runs.max(refused_runs.unwrap_or(0));
        // Understated asks one run more than stood, and only while the pool
        // can stand one more: past that the need asked chunks no pool of
        // this size holds, and the head waited for good with room it fits
        // alone in (the seeded liveness test: three runs of three in a pool
        // of eight). There it is a plain short head, going again once the
        // map has moved past its refusal.
        let max_runs = self.port.state_chunks() / run.max(1);
        // Only a refusal that saw runs of its need standing understates the
        // need. One the map shows runs for while the refusal saw none names
        // a room the map does not read (the unpark's own takes, chunks its
        // serve opened first): grown on it, the need would grow a run a
        // refusal with nothing issued.
        let understated = refused_saw && stood < max_runs;
        let pieces = if understated { pieces.max(stood.saturating_add(1)) } else { pieces };
        let updated = super::proc::PendingBind {
            singles: 0,
            run,
            pieces,
            pages: self.port.chunks_as_pages(run.saturating_mul(pieces)),
        };
        self.stats
            .restore_roles_short
            .fetch_add(1, Ordering::Relaxed);
        // The same refusal again on the same room: each send re-runs the
        // pass (the head's own re-send moves the generation), so no pass
        // would ever settle and the invariant never see it. A repeat is a
        // settled stall: counted, a
        // strict test fails, and it is read as the placement's mismatch
        // (no race refused it twice on one room): held from a send until
        // the room moves, its need planned on chunks only moves empty.
        let room = ((run, pieces), free_chunks, refused_room);
        let repeat = self.with_inner(|inner| {
            inner.procs.get_mut(&pid).is_some_and(|proc| proc.last_refusal.replace(room) == Some(room))
        });
        if repeat {
            self.stats.stuck_settles.fetch_add(1, Ordering::Relaxed);
            self.stats.refusal_repeats.fetch_add(1, Ordering::Relaxed);
            #[cfg(test)]
            if super::plan::PROGRESS_STRICT.with(std::cell::Cell::get) {
                panic!(
                    "progress invariant: restore {pid} refused again on the same room \
                     (need {:?}, free chunks {free_chunks}, free runs {refused_room})",
                    (run, pieces)
                );
            }
        }
        self.with_inner(|inner| {
            let quiet = repeat || inner.nothing_in_flight_but(pid);
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.roles_short = Some(updated);
                // The pending bind keeps its own runs: written back with its
                // unpark's folded in, the next refusal folded them in again.
                proc.roles_free_retry = None;
                proc.roles_short_gives = Some(gives);
                proc.roles_short_runs = Some((need.0, refused_room));
                proc.roles_short_free = Some(free_chunks);
                // Its own restore was in flight; anyone else's?
                proc.roles_short_quiet = quiet;
                // Read below, once it is queued again: until then, as if
                // its placement stood it (the stricter key), so a pass the
                // requeue runs sends it on no less.
                proc.roles_short_stood = Some(true);
                proc.roles_refused = Some(need);
                // Refused against runs the map shows: the census names the
                // mismatch (a room the map does not read).
                proc.roles_short_mismatch =
                    (map_shows && !refused_saw).then_some((map_runs, refused_runs));
                if understated {
                    // Goes again only once more runs of the updated need
                    // stand than its refusal saw.
                    proc.roles_free_short = Some((run, pieces));
                    proc.roles_free_short_at = stood;
                } else {
                    proc.roles_free_short = None;
                }
            }
        });
        self.requeue_restore(pid);
        // The store's own placement of its whole need, its serve's state
        // laid first, as the refusal leaves the pool.
        let queued = self.with_inner(|inner| {
            let bind = inner
                .procs
                .get(&pid)
                .and_then(|proc| proc.bind_room)
                .map(|need| need.entries())
                .unwrap_or_default();
            inner.queue.iter().find_map(|(key, waiter)| {
                (waiter.pid == pid && matches!(waiter.kind, WaitKind::Restore { .. }))
                    .then(|| (*key, waiter.rs_need(), bind.clone(), inner.restore_kv_missing(*key)))
            })
        });
        if let Some((key, state, bind, kv)) = queued {
            let stood = self
                .head_room(key)
                .is_some_and(|need| self.port.restore_stands(pid, &need, state, kv, &bind));
            self.with_inner(|inner| {
                if let Some(proc) = inner.procs.get_mut(&pid) {
                    proc.roles_short_stood = Some(stood);
                }
            });
        }
        // A need past the whole device pool: its gate is told, to refuse it
        // by name ([`Self::restore_never_fits`]) rather than wait.
        if self.restore_never_fits_now(pid)
            && let Some(signal) = self.with_inner(|inner| inner.procs.get(&pid).map(|proc| proc.signal.clone()))
        {
            signal.notify_waiters();
        }
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
        self.port.release_restore(pid);
        // Queued at what it has off the device, read now: its kv pages and
        // its state. Queued at nothing, the planner's need and placement
        // would read no state for it until its next serve re-quoted it,
        // stand a need its serve's state then breaks, and send it again,
        // refused, round after round.
        let (model, engine) = self.port.locus();
        let demand = Demand {
            kv_pages: super::holdings::swapped_page_count(pid, model, engine),
            rs: super::holdings::swapped_rs_units(pid, model, engine),
        };
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
            inner.enqueue(pid, WaitKind::Restore { demand });
        });
        self.poke();
    }

    fn fail_restore_loud(self: &Arc<Self>, pid: ProcessId, reason: &str) {
        let reason = format!("KV restore failed: {reason}");
        // The engine's scheduler gone: the worker is shutting down, and the
        // restore was cut by it, no failure of the process's.
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
