use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::store::kv::page_table::ReclaimQuote;

use super::error::{PlannerError, StarveCause};
use super::grant::Demand;
use super::holdings::{held_page_count, held_rs_units};
use super::proc::Residency;
use super::wait::{EntryKey, WaitKind};
use super::{ProcessId, ResidencyPlanner};

impl ResidencyPlanner {
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
        // So does a process whose own pages and parked asks together are
        // past the pool: a restart asks for the same again.
        if self.check_overcommit() {
            return;
        }
        let Some(victim_key) = self.pick_starvation_victim(cause) else {
            return;
        };
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
            let restarting = crate::inferlet::process::request_restart(victim_pid);
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

    /// [`Self::check_hog`] over a process's concurrent asks, not one: what
    /// it holds and what all its parked asks want, past the pool, can be met
    /// by no eviction of others (the pool is short even with them gone, as
    /// [`PlannerError::PastPool`] is for one ask) and by no restart (it
    /// replays the same asks into the same wall, forever). On a shared pool
    /// its state counts in the kv pages its chunks would hold; on pools of
    /// their own, state is weighed against the state pool. Its parked asks
    /// fail, naming the numbers. Which of several processes yields where
    /// one could fit stays the policy's: this fires only where none fits.
    pub(super) fn check_overcommit(self: &Arc<Self>) -> bool {
        if self.transfers_in_flight() {
            return false;
        }
        let asking: Vec<(ProcessId, u32, Demand)> = self.with_inner(|inner| {
            let mut asking: Vec<(ProcessId, u32, Demand)> = Vec::new();
            for waiter in inner.queue.values() {
                let WaitKind::Allocation {
                    demand,
                    outcome: None,
                    yielded: false,
                    ..
                } = &waiter.kind
                else {
                    continue;
                };
                if demand.is_zero() {
                    continue;
                }
                match asking.iter_mut().find(|(pid, ..)| *pid == waiter.pid) {
                    Some((_, asks, sum)) => {
                        *asks += 1;
                        sum.kv_pages = sum.kv_pages.saturating_add(demand.kv_pages);
                        sum.rs = sum.rs.add(demand.rs);
                    }
                    None => asking.push((waiter.pid, 1, *demand)),
                }
            }
            asking
        });
        let (model, engine) = self.port.locus();
        let (_, total) = self.port.device_stats();
        let (_, rs_total) = self.port.rs_stats();
        let shared = self.port.room().is_some();
        let Some((pid, error)) = asking.into_iter().find_map(|(pid, asks, need)| {
            let held = held_page_count(pid, model, engine);
            let held_rs = held_rs_units(pid, model, engine);
            let state = if shared {
                self.port.rs_as_pages(held_rs.add(need.rs))
            } else {
                0
            };
            let pages = held.saturating_add(need.kv_pages).saturating_add(state);
            if pages > total {
                return Some((
                    pid,
                    PlannerError::Overcommit {
                        asks,
                        need: need.kv_pages,
                        held,
                        state,
                        total,
                    },
                ));
            }
            (!shared && !need.rs.is_zero() && !rs_total.covers(held_rs.add(need.rs))).then_some((
                pid,
                PlannerError::RsHog {
                    need: need.rs,
                    held: held_rs,
                    total: rs_total,
                },
            ))
        }) else {
            return false;
        };
        let notifies: Vec<_> = self.with_inner(|inner| {
            inner
                .queue
                .values_mut()
                .filter(|waiter| waiter.pid == pid)
                .filter_map(|waiter| match &mut waiter.kind {
                    WaitKind::Allocation {
                        notify,
                        outcome: outcome @ None,
                        yielded: false,
                        ..
                    } => {
                        *outcome = Some(Err(error));
                        Some(notify.clone())
                    }
                    _ => None,
                })
                .collect()
        });
        if notifies.is_empty() {
            return false;
        }
        self.stats.hog_failures.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(%pid, %error, "planner: a process's own asks are past the pool — failing them");
        for notify in notifies {
            notify.notify_waiters();
        }
        true
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
                ..
            } = &mut waiter.kind
            else {
                return None;
            };
            let need = demand.kv_pages;
            if outcome.is_some() || need.saturating_add(held) <= total {
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
