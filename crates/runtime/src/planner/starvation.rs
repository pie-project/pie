use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::store::kv::page_table::ReclaimQuote;

use super::error::{PlannerError, StarveCause};
use super::holdings::{held_page_count, held_rs_units};
use super::proc::Residency;
use super::wait::{EntryKey, WaitKind};
use super::{ProcessId, ResidencyPlanner};

impl ResidencyPlanner {
    /// Every tier full, the stall's head with no move left (its victims'
    /// moves found no room below, nothing in flight): the would-be victim,
    /// the lowest of the residents but the head in the policy's order
    /// (priority, then first come), fails by name with every tier's room
    /// ([`PlannerError::TiersFull`]), its room freed for the others; with
    /// none resident, the lowest of the parked (their room below frees for
    /// the moves). Over-demand with room below waits and moves; only no
    /// room anywhere ends a process (the simulation's Metal seed 74 and
    /// field seed 401: the disk full of the evicted, every candidate's move
    /// refused below, all processes stalled for good). `chosen`: the victim
    /// already named (an eviction whose copy below found the tier full).
    /// Whether one was failed.
    pub(super) fn fail_tiers_full(self: &Arc<Self>, chosen: Option<ProcessId>) -> bool {
        let tiers = PlannerError::TiersFull {
            device: self.port.device_stats(),
            host: self.port.host_stats(),
            disk: self.port.disk_stats(),
        };
        // Who holds room a failure frees: device kv or guest state. A
        // process holding none (one not admitted yet, at a gate before its
        // first ask) frees nothing failed, and its wait is no planner wait
        // a failure reaches (the simulation's Metal seed 42: one failed at
        // admission never departed, its kill in flight for good).
        let (model, engine) = self.port.locus();
        let pids: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
        let holding: std::collections::HashSet<ProcessId> = pids
            .into_iter()
            .filter(|&pid| {
                held_page_count(pid, model, engine)
                    > super::holdings::swapped_page_count(pid, model, engine)
                    || self.port.guest_pages_of(pid) > 0
            })
            .collect();
        let Some((victim, head, signal, notify)) = self.with_inner(|inner| {
            let head = inner.unmet_head().map(|(_, head)| head.pid);
            let lowest = |resident: bool| {
                inner
                    .procs
                    .iter()
                    .filter(|(pid, proc)| {
                        Some(**pid) != head
                            && proc.admitted
                            && holding.contains(*pid)
                            && !inner.killing.contains(*pid)
                            && (proc.state == Residency::Resident) == resident
                    })
                    .max_by_key(|(pid, proc)| (proc.rank(), proc.seq, **pid))
                    .map(|(pid, _)| *pid)
            };
            let victim = chosen
                .filter(|pid| !inner.killing.contains(pid))
                .or_else(|| lowest(true))
                .or_else(|| lowest(false))?;
            inner.killing.insert(victim);
            inner.failed.insert(victim, tiers);
            // Each of its waits ends with the failure: an allocation's
            // outcome, a bind's or a restore's wait gone from the queue.
            let mut notify = Vec::new();
            let keys: Vec<EntryKey> = inner
                .queue
                .iter()
                .filter(|(_, waiter)| waiter.pid == victim)
                .map(|(&key, _)| key)
                .collect();
            for key in keys {
                let Some(waiter) = inner.queue.get_mut(&key) else {
                    continue;
                };
                match &mut waiter.kind {
                    WaitKind::Allocation { outcome, notify: wake, .. } if outcome.is_none() => {
                        *outcome = Some(Err(tiers));
                        notify.push(wake.clone());
                    }
                    WaitKind::BindRoom { notify: wake, .. } => {
                        notify.push(wake.clone());
                        inner.queue.remove(&key);
                    }
                    WaitKind::Restore { .. } => {
                        inner.queue.remove(&key);
                    }
                    WaitKind::Allocation { .. } => {}
                }
            }
            let signal = inner.procs.get(&victim).map(|proc| proc.signal.clone());
            Some((victim, head, signal, notify))
        }) else {
            return false;
        };
        let reason = format!("planner: {tiers}");
        tracing::warn!(pid = %victim, head = ?head, %reason, "planner: a process fails: every tier is full");
        self.stats.starvations.fetch_add(1, Ordering::Relaxed);
        crate::scheduler::worker::notify_process_resume(victim);
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        for wake in notify {
            wake.notify_waiters();
        }
        crate::inferlet::process::terminate_by(victim, Err(reason), crate::policy::By::Host);
        self.poke();
        true
    }

    pub(super) fn check_rs_starvation(self: &Arc<Self>, key: EntryKey) {
        if !self.is_wedged() || self.waits_on_policy() {
            return;
        }
        let Some((pid, need)) = self.with_inner(|inner| {
            inner
                .queue
                .get(&key)
                .map(|waiter| (waiter.pid, waiter.rs_need()))
        }) else {
            return;
        };
        let (free, total) = self.port.rs_stats();
        if free.covers(need) {
            self.poke();
            return;
        }
        if !self.is_wedged() {
            return;
        }
        // What the head itself holds is not coming back while it waits, so
        // past that the pool is too small for it; short of that, the slots
        // are held by processes no suspend can move, and a restart frees them.
        let (model, engine) = self.port.locus();
        let held = held_rs_units(pid, model, engine);
        let restore = self.with_inner(|inner| {
            inner
                .queue
                .get(&key)
                .is_some_and(|waiter| matches!(waiter.kind, WaitKind::Restore { .. }))
        });
        if restore || total.covers(need.add(held)) {
            self.check_starvation(StarveCause::NoRsSlots);
            return;
        }
        let notify = self.with_inner(|inner| {
            match inner.unmet_head() {
                Some((head, _)) if head == key => {}
                _ => return None,
            }
            let waiter = inner.queue.get_mut(&key)?;
            let WaitKind::Allocation {
                notify, outcome, ..
            } = &mut waiter.kind
            else {
                return None;
            };
            if outcome.is_some() {
                return None;
            }
            *outcome = Some(Err(PlannerError::RsHog { need, held, total }));
            Some(notify.clone())
        });
        if let Some(notify) = notify {
            self.stats.hog_failures.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                %need,
                %free,
                %held,
                %total,
                "planner: RS ask past the pool — failing the head"
            );
            notify.notify_waiters();
        }
    }

    pub(super) fn check_starvation(self: &Arc<Self>, cause: StarveCause) {
        if !self.is_wedged() || self.waits_on_policy() {
            return;
        }
        // Idle cache on its way to the host is neither dropped nor swept
        // while it moves, and frees its device pages when it lands (which
        // pokes the planner): the pool is not starved, it is waiting.
        if self.demoting.load(Ordering::Acquire) || self.sweeping.load(Ordering::Acquire) {
            return;
        }
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        if cause == StarveCause::NoEligibleVictim && self.last_resort_evict() {
            return;
        }
        if !self.is_wedged() {
            return;
        }
        if self.salvage_free_pages() {
            self.poke();
            return;
        }
        if self.serve_from_hoard() {
            return;
        }
        let (free, total) = self.port.device_stats();
        if self.with_inner(|inner| inner.kill_in_flight()) {
            return;
        }
        // A head past the pool with what it holds itself fails: restarting
        // it would only replay it into the same wall.
        if let Some((head, pid)) =
            self.with_inner(|inner| inner.unmet_head().map(|(k, w)| (k, w.pid)))
            && self.check_hog(head, pid)
        {
            return;
        }
        let Some(victim_key) = self.pick_starvation_victim(cause) else {
            return;
        };
        // A victim that alone holds anything in the pool, with nothing of
        // it running, replays into the same stall: it fails, not restarts.
        let cause = match self.with_inner(|inner| inner.queue.get(&victim_key).map(|w| w.pid)) {
            Some(victim) if self.sole_holder(victim) => StarveCause::SoleHolder,
            _ => cause,
        };
        // The head itself picked, short only of room others hold that no
        // move frees now (no eligible victim), its need fitting the pool
        // alone: it waits (the user's ruling: no waiter that fits alone
        // gives up). The holders' completions, a room change or an arrival
        // plan again; the stall is named meanwhile (`dump_stall`). Failed
        // here, a rise that fit was refused once its holders' moves found
        // nothing to move.
        // So too any other waiter picked: one that fits the pool alone is
        // never failed for another's room (seed 3243: a decoder's one-page
        // ask failed for a head behind a full disk, the pool able to serve
        // both in turn).
        let victim_fits_alone = self.with_inner(|inner| {
            inner
                .queue
                .get(&victim_key)
                .map(|victim| victim.kv_need() <= total)
        });
        if cause != StarveCause::SoleHolder && victim_fits_alone == Some(true) {
            return;
        }
        let (rs_free, rs_total) = self.port.rs_stats();
        let notify = self.with_inner(|inner| {
            let (_, head) = inner.unmet_head()?;
            // A head the pools already cover is served next, so a kill would
            // free nothing it lacks. One short of state alone is not covered
            // by the pages it has.
            let servable =
                (inner.accum.len() as u32) >= head.kv_need() && rs_free.covers(head.rs_need());
            if servable || !inner.evicting.is_empty() {
                return None;
            }
            let victim = inner.queue.get_mut(&victim_key)?;
            let victim_pid = victim.pid;
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                ..
            } = &mut victim.kind
            else {
                return None;
            };
            if outcome.is_some() {
                return None;
            }
            // A shared pool routes a state shortfall as kv pages to park: an
            // ask whose pages stand free lacked the state, and says so.
            let state = (cause == StarveCause::NoRsSlots || !rs_free.covers(demand.rs))
                .then_some((demand.rs, rs_free, rs_total));
            *outcome = Some(Err(PlannerError::Starved {
                need: demand.kv_pages,
                free,
                total,
                cause,
                state,
            }));
            let notify = notify.clone();
            inner.killing.insert(victim_pid);
            Some((notify, victim_pid))
        });
        if let Some((notify, victim_pid)) = notify {
            self.stats.starvations.fetch_add(1, Ordering::Relaxed);
            let restarting = cause != StarveCause::SoleHolder
                && crate::inferlet::process::request_restart(victim_pid);
            if restarting {
                self.stats
                    .starvation_restarts
                    .fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    cause = ?cause,
                    pid = %victim_pid,
                    "planner: pool starved — reclaiming a restartable process, its work is re-queued"
                );
            } else {
                tracing::warn!(
                    cause = ?cause,
                    pid = %victim_pid,
                    "planner: pool starved with no reclaim path — failing the youngest parked ask that holds pages"
                );
            }
            notify.notify_waiters();
        }
    }

    /// `pid` holds all that is held in the pool: no other process holds kv
    /// pages or state or guest state on the device
    /// ([`super::PoolPort::holds_device_room`]: its instances' scratch, its
    /// parked role slots lying there), none is on its way back, and no idle
    /// cache could go.
    /// Stalled (nothing of it in flight), it is stuck on what it holds
    /// itself, and a restart frees only that to ask for the same again.
    pub(super) fn sole_holder(&self, pid: ProcessId) -> bool {
        let (model, engine) = self.port.locus();
        let (others, quiet) = self.with_inner(|inner| {
            let others: Vec<ProcessId> = inner
                .procs
                .keys()
                .filter(|other| **other != pid)
                .copied()
                .collect();
            (others, inner.evicting.is_empty())
        });
        quiet
            && self.port.reclaimable_idle() == 0
            && others.into_iter().all(|other| {
                held_page_count(other, model, engine) == 0
                    && held_rs_units(other, model, engine).is_zero()
                    && !self.port.holds_device_room(other)
            })
    }

    fn pick_starvation_victim(&self, cause: StarveCause) -> Option<EntryKey> {
        let candidates: Vec<(EntryKey, ProcessId)> = self.with_inner(|inner| {
            inner
                .queue
                .iter()
                .rev()
                .filter(|(_, waiter)| {
                    matches!(&waiter.kind, WaitKind::Allocation { outcome: None, .. })
                })
                .map(|(key, waiter)| (*key, waiter.pid))
                .collect()
        });
        let fallback = candidates.first().map(|(key, _)| *key);
        let (model, engine) = self.port.locus();
        let mut first_holder = None;
        for restartable_only in [true, false] {
            let selected: Vec<(EntryKey, ProcessId)> = candidates
                .iter()
                .filter(|(_, pid)| {
                    !restartable_only || crate::inferlet::process::is_restartable(*pid)
                })
                .copied()
                .collect();
            if selected.is_empty() {
                continue;
            }
            let pids: Vec<ProcessId> = selected.iter().map(|(_, pid)| *pid).collect();
            let quotes =
                crate::inferlet::process::residency::kv_reclaim_quotes(&pids, model, engine, 1);
            for ((key, pid), quote) in selected.iter().zip(quotes) {
                // A restart frees what the stall lacks: pages, or the state
                // slots (windows included) a stuck restore waits on.
                let holds = match cause {
                    StarveCause::NoRsSlots => !held_rs_units(*pid, model, engine).is_zero(),
                    _ => matches!(quote, Some(ReclaimQuote::Pages(pages)) if pages > 0),
                };
                if holds {
                    if restartable_only {
                        return Some(*key);
                    }
                    first_holder.get_or_insert(*key);
                    break;
                }
            }
        }
        first_holder.or(fallback)
    }

    fn transfers_in_flight(&self) -> bool {
        self.with_inner(|inner| {
            !inner.evicting.is_empty()
                || inner
                    .procs
                    .values()
                    .any(|proc| proc.state == Residency::Restoring)
        })
    }

    pub(super) fn check_hog(self: &Arc<Self>, head_key: EntryKey, head_pid: ProcessId) -> bool {
        if self.transfers_in_flight() {
            return false;
        }
        let (model, engine) = self.port.locus();
        let held = held_page_count(head_pid, model, engine);
        let (_, total) = self.port.device_stats();
        let notify = self.with_inner(|inner| {
            let waiter = inner.queue.get_mut(&head_key)?;
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                rise,
                ..
            } = &mut waiter.kind
            else {
                return None;
            };
            let need = demand.kv_pages;
            // A rise moves the requester's own pages, holding none past its
            // grant: what one process holds the load seats beside the
            // largest rung, so a rise is no hog.
            if outcome.is_some() || *rise || need.saturating_add(held) <= total {
                return None;
            }
            *outcome = Some(Err(PlannerError::Hog { need, held, total }));
            Some(notify.clone())
        });
        let Some(notify) = notify else {
            return false;
        };
        self.stats.hog_failures.fetch_add(1, Ordering::Relaxed);
        notify.notify_waiters();
        true
    }
}
