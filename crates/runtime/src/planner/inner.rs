use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::ProcessId;
use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::proc::{Proc, Residency};
use super::rank::Rank;
use super::reclaim::Reclaim;
use super::wait::{EntryKey, WaitKind, Waiter};

/// The most of the window lists the seated lanes' growth holds back from a prompt
/// waiting to be seated: a thirty-second, as a small pool seats few lanes
/// and serves their growth from what the others let go.
pub(super) const RESERVE_SHARE: u32 = 32;

#[derive(Default)]
pub(super) struct Inner {
    /// The process suspended for a bind whose restore was served and that
    /// has not fired since: one at a time. Restored together, each took its
    /// items back and every bind came up short against the others' (the M1
    /// stress: ten processes suspended, restored and short in turn, 190
    /// times each); restored alone, a bind competes only with the
    /// residents.
    pub(super) bind_restoring: Option<ProcessId>,
    pub(super) next_seq: u64,
    next_id: u64,
    pub(super) procs: HashMap<ProcessId, Proc>,
    pub(super) queue: BTreeMap<EntryKey, Waiter>,
    pub(super) accum: DevicePageReservation,
    /// Pages on their way back: a suspend's in flight, keyed by its process,
    /// or a departed process's still held by its working sets, keyed by the
    /// process gone, until its last [`super::Departure`] hold drops.
    pub(super) evicting: HashMap<ProcessId, Reclaim>,
    pub(super) completion_credit: u32,
    pub(super) host_swap_blocked: HashSet<ProcessId>,
    pub(super) prepare_blocked: HashSet<ProcessId>,
    pub(super) killing: HashSet<ProcessId>,
    pub(super) admitting: BTreeMap<(Rank, ProcessId), Arc<Notify>>,
    pub(super) to_end: super::to_end::ToEnd,
}

impl Inner {
    pub(super) fn kill_in_flight(&self) -> bool {
        self.killing.iter().any(|pid| {
            self.procs.contains_key(pid)
                && !self
                    .queue
                    .values()
                    .any(|waiter| waiter.pid == *pid && waiter.is_unmet())
        })
    }

    /// `pid` is resident again: the youngest resident, the first victim
    /// past the policy's keys ([`super::rank::VictimOrder`]).
    pub(super) fn came_back(&mut self, pid: ProcessId) {
        let came = self.next_seq;
        if let Some(proc) = self.procs.get_mut(&pid) {
            proc.came = came;
            self.next_seq += 1;
        }
    }

    /// The pages that, moved out whole, free the runs of chunks `waiter`
    /// asks past its kv: a bind short of guest-state room, or the restore
    /// of a process such a bind suspended ([`super::proc::Proc::bind_room`]).
    pub(super) fn room_pages(&self, waiter: &super::wait::Waiter) -> u32 {
        let pending = match waiter.kind {
            // A bind its suspension waits for, or role slots its last
            // unpark was short of: pages that, moved out, free their chunks.
            WaitKind::Restore { .. } => self.procs.get(&waiter.pid).map_or(0, |proc| {
                let bind = proc.bind_room.map_or(0, |need| need.pages);
                let roles = proc.roles_short.map_or(0, |need| need.pages);
                bind.max(roles)
            }),
            _ => 0,
        };
        waiter.room_pages().max(pending)
    }

    pub(super) fn settle_eviction(&mut self, pid: ProcessId) {
        self.evicting.remove(&pid);
    }

    /// What the evictions in flight are expected to free.
    pub(super) fn expected_reclaim(&self) -> Reclaim {
        self.evicting
            .values()
            .fold(Reclaim::default(), |sum, &mark| sum.add(mark))
    }

    /// The processes that cannot fire now: off the device (a handle at
    /// rest aside), or waiting in the queue for pages, room for a bind, or
    /// their restore. What an admission gate claims nothing for
    /// (`inferlet::process::ordered`): they claim again once un-parked.
    pub(super) fn parked(&self) -> impl Iterator<Item = ProcessId> + '_ {
        let off = self
            .procs
            .iter()
            .filter(|(_, proc)| off_device(proc))
            .map(|(pid, _)| *pid);
        off.chain(self.queued_on_device())
    }

    /// The processes waiting in the queue that are not off the device,
    /// each once.
    fn queued_on_device(&self) -> Vec<ProcessId> {
        let mut waiting: Vec<ProcessId> = self
            .queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .map(|waiter| waiter.pid)
            .filter(|pid| self.procs.get(pid).is_none_or(|proc| !off_device(proc)))
            .collect();
        waiting.sort_unstable();
        waiting.dedup();
        waiting
    }

    /// A digest of [`Self::parked`]: it changes when the set does.
    pub(super) fn parked_digest(&self) -> u64 {
        self.parked().fold(0u64, |digest, pid| {
            let bits = pid.as_u128();
            digest ^ (bits as u64) ^ ((bits >> 64) as u64)
        })
    }

    pub(super) fn nonresident_count(&self) -> usize {
        let mut n = 0;
        for proc in self.procs.values() {
            let resident = proc.state == Residency::Resident;
            // A handle at rest off the device is the point of parking it.
            if !resident && !proc.holder {
                n += 1;
            }
            proc.resident.store(resident, Ordering::Release);
        }
        n
    }

    pub(super) fn preempted_floor(&self) -> Option<Rank> {
        self.procs
            .iter()
            .filter(|(pid, proc)| {
                (proc.state != Residency::Resident && proc.parked.is_none())
                    || proc.respawn
                    || self.killing.contains(pid)
            })
            .map(|(_, proc)| proc.rank())
            .min()
    }

    pub(super) fn admit_unpreempted(&mut self) {
        if self.admitting.is_empty() {
            return;
        }
        let floor = self.preempted_floor();
        let (go, blocked): (BTreeMap<_, _>, BTreeMap<_, _>) = std::mem::take(&mut self.admitting)
            .into_iter()
            .partition(|((rank, _), _)| floor.is_none_or(|floor| *rank <= floor));
        self.admitting = blocked;
        for notify in go.into_values() {
            notify.notify_one();
        }
    }

    /// What an unseated claim on `need` leaves free for seated lanes to grow into
    /// (`growth` each), so seating a prompt never leaves one short at its next page.
    pub(super) fn seating_reserve(
        &self,
        pid: ProcessId,
        growth: RsUnits,
        need: RsUnits,
        capacity: RsUnits,
    ) -> RsUnits {
        if need.is_zero()
            || growth.is_zero()
            || self.procs.get(&pid).is_none_or(|proc| proc.seated)
            || self.fleet_stalled()
        {
            return RsUnits::default();
        }
        let seated = u32::try_from(
            self.procs
                .values()
                .filter(|proc| proc.seated && !proc.idle && proc.state == Residency::Resident)
                .count(),
        )
        .unwrap_or(u32::MAX);
        RsUnits {
            slots: growth
                .slots
                .saturating_mul(seated)
                .min(capacity.slots / RESERVE_SHARE),
            pages: growth
                .pages
                .saturating_mul(seated)
                .min(capacity.pages / RESERVE_SHARE),
            ..RsUnits::default()
        }
    }

    /// A seated lane's growth takes free pages at once rather than parking behind
    /// claims served after it anyway (a waiting prompt holding nothing); a
    /// higher-ranked seated waiter or preempted process still goes first.
    pub(super) fn growth_first(&self, pid: ProcessId) -> bool {
        let Some(proc) = self.procs.get(&pid) else {
            return false;
        };
        let rank = proc.rank();
        proc.state == Residency::Resident
            && !proc.respawn
            && !self.killing.contains(&pid)
            && self.queue.iter().all(|(&(waiter_rank, _), waiter)| {
                waiter.pid != pid
                    && (waiter_rank > rank
                        || !waiter.is_unmet()
                        || self.procs.get(&waiter.pid).is_some_and(|w| !w.seated))
            })
            && self.procs.iter().all(|(other, proc)| {
                proc.rank() > rank
                    || (proc.state == Residency::Resident
                        && !proc.respawn
                        && !self.killing.contains(other))
            })
    }

    pub(super) fn enqueue(&mut self, pid: ProcessId, kind: WaitKind) -> EntryKey {
        let key = (self.procs[&pid].rank(), self.next_id);
        self.next_id += 1;
        self.queue.insert(key, Waiter { pid, kind });
        key
    }

    /// Wherever a reschedule has since moved it.
    pub(super) fn locate(&self, key: EntryKey) -> Option<EntryKey> {
        if self.queue.contains_key(&key) {
            return Some(key);
        }
        self.queue.keys().find(|(_, id)| *id == key.1).copied()
    }

    pub(super) fn reschedule(&mut self, order_of: impl Fn(&ProcessId) -> (i64, bool)) {
        for (pid, proc) in self.procs.iter_mut().filter(|(_, proc)| !proc.holder) {
            (proc.key, proc.held) = order_of(pid);
        }
        let held_restores: Vec<EntryKey> = self
            .queue
            .iter()
            .filter(|(_, w)| matches!(w.kind, WaitKind::Restore { .. }))
            .filter(|(_, w)| self.procs.get(&w.pid).is_some_and(|p| p.held))
            .map(|(key, _)| *key)
            .collect();
        for key in held_restores {
            if let Some(Waiter {
                pid,
                kind: WaitKind::Restore { demand },
            }) = self.queue.remove(&key)
                && let Some(proc) = self.procs.get_mut(&pid)
            {
                proc.parked = Some(demand);
            }
        }
        let released: Vec<(ProcessId, Demand)> = self
            .procs
            .iter_mut()
            .filter(|(_, p)| !p.held && !p.idle && !p.seatless && p.state == Residency::Evicted)
            .filter_map(|(pid, p)| Some((*pid, p.parked.take()?)))
            .collect();
        for (pid, demand) in released {
            self.enqueue(pid, WaitKind::Restore { demand });
        }
        let procs = &self.procs;
        let rank = |pid: &ProcessId, was: Rank| procs.get(pid).map_or(was, Proc::rank);
        self.queue = std::mem::take(&mut self.queue)
            .into_iter()
            .map(|((was, id), waiter)| ((rank(&waiter.pid, was), id), waiter))
            .collect();
        self.admitting = std::mem::take(&mut self.admitting)
            .into_iter()
            .map(|((was, pid), notify)| ((rank(&pid, was), pid), notify))
            .collect();
    }

    pub(super) fn unmet_head(&self) -> Option<(EntryKey, &Waiter)> {
        let mut oldest_restore: Option<EntryKey> = None;
        let mut allocation: Option<EntryKey> = None;
        for (&key, waiter) in &self.queue {
            if !waiter.is_unmet() {
                continue;
            }
            match &waiter.kind {
                // A bind short of room asks as an admission does.
                WaitKind::Allocation { .. } | WaitKind::BindRoom { .. } => {
                    allocation = Some(key);
                    break;
                }
                WaitKind::Restore { .. } => {
                    oldest_restore.get_or_insert(key);
                }
            }
        }
        let head = match (allocation, oldest_restore) {
            (Some(allocation), Some(restore)) => {
                // An older restore comes back ahead of new work: the
                // admission of a prompt not yet seated waits behind it
                // (Mac Pro's M5: processes evicted in the ramp waited
                // behind every later arrival, then froze the tail). A seated
                // process's own growth keeps its turn, or the decoders whose
                // completions free the restore's room would wait on it.
                let admits_new = self
                    .queue
                    .get(&allocation)
                    .and_then(|waiter| self.procs.get(&waiter.pid))
                    .is_some_and(|proc| !proc.seated);
                if restore < allocation
                    && (admits_new || self.fleet_stalled() || self.eviction_unfundable())
                {
                    restore
                } else {
                    allocation
                }
            }
            (Some(allocation), None) => allocation,
            (None, restore) => restore?,
        };
        self.queue
            .get_key_value(&head)
            .map(|(&key, waiter)| (key, waiter))
    }

    pub(super) fn queued_quantum(&self) -> u32 {
        self.queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .filter(|waiter| {
                matches!(&waiter.kind, WaitKind::Allocation { demand, .. } if demand.kv_pages == 1)
            })
            .count() as u32
    }

    pub(super) fn begin_eviction(&mut self, pid: ProcessId, expected: Reclaim) -> Vec<Arc<Notify>> {
        if let Some(proc) = self.procs.get_mut(&pid) {
            proc.state = Residency::Evicting;
        }
        self.evicting.insert(pid, expected);
        let mut wake = Vec::new();
        for waiter in self.queue.values_mut().filter(|w| w.pid == pid) {
            if let WaitKind::Allocation {
                notify, yielded, ..
            } = &mut waiter.kind
            {
                *yielded = true;
                wake.push(notify.clone());
            }
        }
        wake
    }

    pub(super) fn eviction_unfundable(&self) -> bool {
        !self.host_swap_blocked.is_empty()
    }

    pub(super) fn burst_shortfall(&self) -> u32 {
        let mut supply = self.accum.len() as u32;
        let mut extra = 0u32;
        for waiter in self.queue.values() {
            if !waiter.is_unmet() {
                continue;
            }
            let WaitKind::Allocation { demand, .. } = &waiter.kind else {
                continue;
            };
            if !demand.rs.is_zero() {
                break;
            }
            let need = demand.kv_pages;
            let covered = need.min(supply);
            supply -= covered;
            extra = extra.saturating_add(need - covered);
        }
        extra
    }

    pub(super) fn serve_burst(&mut self) -> Vec<(EntryKey, u32, Arc<Notify>)> {
        let mut wake = Vec::new();
        let Some((head, waiter)) = self.unmet_head() else {
            return wake;
        };
        if !matches!(waiter.kind, WaitKind::Allocation { .. }) {
            return wake;
        }
        let Inner { queue, accum, .. } = self;
        for (&key, waiter) in queue.range_mut(head..) {
            if !waiter.is_unmet() {
                continue;
            }
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                ..
            } = &mut waiter.kind
            else {
                continue;
            };
            let demand = *demand;
            if !demand.rs.is_zero() {
                break;
            }
            if (accum.len() as u32) < demand.kv_pages {
                break;
            }
            let kv = accum.donate(demand.kv_pages);
            debug_assert!(outcome.is_none(), "an unmet waiter carries no outcome");
            *outcome = Some(Ok(AllocationGrant::new(kv, RsSlotReservation::empty())));
            wake.push((key, demand.kv_pages, notify.clone()));
        }
        wake
    }

    /// The pages the resident lanes' next steps take: one each process
    /// running (admitted, resident, not idle). A restore leaves this much of
    /// the device free, so bringing one back does not suspend another at
    /// the next page boundary.
    pub(super) fn growth_reserve(&self) -> u32 {
        self.procs
            .values()
            .filter(|proc| proc.admitted && !proc.idle && proc.state == Residency::Resident)
            .count() as u32
    }

    /// Whether the restore of `pid`, the head, goes now (`spare`: device
    /// pages free past what the head gathered). A restore goes back once a
    /// resident process has finished (its credit), or the fleet can do
    /// nothing else, or the device has room for it and still a page for
    /// each resident lane's next one: suspended processes come back as
    /// soon as room appears, every one it holds at once, not one a
    /// completion. One whose role slots found no room last time
    /// ([`super::proc::Proc::roles_short`]) waits while the fleet decodes
    /// (a completion frees room; funding it then thrashed), and once the
    /// fleet is stalled (a fleet waiting at binds included) has room made
    /// for its runs: idle cache, a coming grow, then the occupants of the
    /// chunks it could take.
    pub(super) fn restore_gate(&mut self, pid: ProcessId, spare: u32) -> RestoreGate {
        let roles_short = self.procs.get(&pid).and_then(|proc| proc.roles_short);
        if self.completion_credit > 0 {
            self.completion_credit -= 1;
            return RestoreGate::Serve;
        }
        let stalled = self.fleet_stalled();
        // Short of runs for its roles: it goes again on runs that stand
        // free whatever the fleet does (the plan looks, outside this lock);
        // room is made for it by moving others only once the fleet is
        // stalled or waits at binds. Waiting for a completion while its
        // runs stood free, it waited on an event that need not come (Mac
        // Pro's beam mix: 2 runs of 4 short, 440 such runs free, flat for
        // 191 s with every thread parked).
        if let Some(need) = roles_short {
            return RestoreGate::Evict {
                need: (need.run, need.pieces),
                fund: stalled || self.fleet_waits_at_binds(),
            };
        }
        if spare < self.growth_reserve() && !stalled {
            return RestoreGate::Wait;
        }
        RestoreGate::Serve
    }

    /// The head restore waiting on room for its roles goes again: room
    /// was made for it ([`Step::EvictForRoles`](super::plan)).
    pub(super) fn roles_retry_head(&mut self) {
        let Some((_, waiter)) = self.unmet_head() else {
            return;
        };
        let pid = waiter.pid;
        if let Some(proc) = self.procs.get_mut(&pid) {
            proc.roles_short = None;
        }
    }

    /// No resident process makes progress save through room: each is
    /// stalled as [`Self::fleet_stalled`] counts it, or waits at a bind for
    /// room (AMD's 7-minute stall: 143 processes at a bind, the restore
    /// head short of role slots never funded). Read only where a restore
    /// short of role slots asks whether to wait for a completion: a bind
    /// waiter is no wedge for the starvation verdict, which may still
    /// move it.
    pub(super) fn fleet_waits_at_binds(&self) -> bool {
        self.evicting.is_empty()
            && self.procs.values().any(|proc| proc.bind_waiting)
            && self.procs.iter().all(|(pid, proc)| match proc.state {
                Residency::Resident => {
                    !proc.admitted
                        || proc.idle
                        || proc.bind_waiting
                        || (self.queue.values().any(|waiter| waiter.pid == *pid) && !proc.firing())
                }
                Residency::Evicting | Residency::Restoring => false,
                Residency::Evicted => true,
            })
    }

    pub(super) fn fleet_stalled(&self) -> bool {
        if !self.evicting.is_empty() {
            return false;
        }
        let mut parked = HashSet::new();
        for waiter in self.queue.values() {
            match &waiter.kind {
                WaitKind::Allocation {
                    outcome: Some(_), ..
                } => return false,
                _ => {
                    parked.insert(waiter.pid);
                }
            }
        }
        self.procs.iter().all(|(pid, proc)| match proc.state {
            // Parked, it is stalled only with no fire of its own in flight:
            // one parked lane beside its peers' running fires waits for them.
            Residency::Resident => {
                !proc.admitted || proc.idle || (parked.contains(pid) && !proc.firing())
            }
            Residency::Evicting | Residency::Restoring => false,
            Residency::Evicted => true,
        })
    }
}

/// What the head restore does now ([`Inner::restore_gate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreGate {
    Serve,
    Wait,
    /// Its role slots are short of `need`, `(run, pieces)` runs of whole
    /// chunks: it goes again where they stand free, and (`fund`) room is
    /// made for it ([`Step::EvictForRoles`](super::plan)).
    Evict { need: (u32, u32), fund: bool },
}

/// Off the device, a handle at rest aside: it cannot fire until restored.
fn off_device(proc: &Proc) -> bool {
    proc.state != Residency::Resident && !proc.holder
}
