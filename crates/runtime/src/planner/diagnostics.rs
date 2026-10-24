use std::sync::atomic::Ordering;

use crate::store::rs::RsUnits;

use super::holdings::held_page_count;
use super::proc::Residency;
use super::wait::WaitKind;
use super::{ProcessId, ResidencyPlanner};

#[derive(Debug, Clone)]
pub struct PlannerQueueEntry {
    pub process_id: String,
    pub spawn_seq: u64,
    pub kind: &'static str,
    pub pages: u32,
    pub rs: RsUnits,
}

const RUNNER_DUMP_CAP: usize = 24;

#[derive(Debug, Clone)]
pub struct PlannerDiagnostics {
    pub device_pages_free: u32,
    pub device_pages_total: u32,
    pub host_slots_free: u32,
    pub host_slots_total: u32,
    pub disk_slots_free: u32,
    pub disk_slots_total: u32,
    pub rs_free: RsUnits,
    pub rs_total: RsUnits,
    pub accumulation: u32,
    pub queue: Vec<PlannerQueueEntry>,
    pub proc_states: [u32; 4],
    pub admitted_procs: u32,
    pub runners: Vec<(u64, u32, bool)>,
    pub parks_total: u64,
    pub serves_total: u64,
    pub evictions_total: u64,
    pub eviction_deferrals_total: u64,
    pub unmet_head_pages: u32,
    pub unmet_head_kind: &'static str,
    pub unmet_queued: u32,
    pub bypassable_entries: u32,
    pub bypassable_pages: u32,
    pub eviction_rollbacks_total: u64,
    pub restores_total: u64,
    pub restore_failures_total: u64,
    pub gate_parks_total: u64,
    pub cancelled_waits_total: u64,
    pub hog_failures_total: u64,
    pub starvations_total: u64,
    pub starvation_restarts_total: u64,
    pub salvages_total: u64,
    pub hoard_bypasses_total: u64,
    pub e6_relaxations_total: u64,
    pub restore_absorb_short_total: u64,
    pub host_swap_exhaustions_total: u64,
    pub host_swap_unblocks_total: u64,
    pub d2h_pages_total: u64,
    pub h2d_pages_total: u64,
    pub disk_write_pages_total: u64,
    pub disk_read_pages_total: u64,
    pub disk_write_us_total: u64,
    pub disk_read_us_total: u64,
    pub d2h_rs_slots_total: u64,
    pub h2d_rs_slots_total: u64,
    pub d2h_copy_us_total: u64,
    pub h2d_copy_us_total: u64,
}

impl ResidencyPlanner {
    pub fn diagnostics(&self) -> PlannerDiagnostics {
        let (device_pages_free, device_pages_total) = self.port.device_stats();
        let (host_slots_free, host_slots_total) = self.port.host_stats();
        let (disk_slots_free, disk_slots_total) = self.port.disk_stats();
        let (rs_free, rs_total) = self.port.rs_stats();
        let inner = self.lock_inner();
        let queue = inner
            .queue
            .iter()
            .map(|(key, waiter)| PlannerQueueEntry {
                process_id: waiter.pid.to_string(),
                spawn_seq: key.0.seq,
                kind: waiter.kind_name(),
                pages: waiter.kv_need(),
                rs: waiter.rs_need(),
            })
            .collect();
        let head = inner.unmet_head();
        let (unmet_head_pages, unmet_head_kind) = head.map_or((0, "-"), |(_, waiter)| {
            (waiter.kv_need(), waiter.kind_name())
        });
        let unmet_queued = inner
            .queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .count() as u32;
        let (bypassable_entries, bypassable_pages) = {
            match head.map(|(key, _)| key.0) {
                None => (0, 0),
                Some(head) => {
                    let mut budget = device_pages_free;
                    let (mut entries, mut pages) = (0u32, 0u32);
                    for (key, waiter) in inner.queue.iter() {
                        if key.0 <= head || !waiter.is_unmet() {
                            continue;
                        }
                        let need = waiter.kv_need();
                        if need <= budget {
                            budget -= need;
                            entries += 1;
                            pages += need;
                        }
                    }
                    (entries, pages)
                }
            }
        };
        let parked: std::collections::HashSet<ProcessId> =
            inner.queue.values().map(|waiter| waiter.pid).collect();
        let mut runner_ids: Vec<(ProcessId, u64, bool)> = inner
            .procs
            .iter()
            .filter(|(pid, proc)| proc.admitted && !parked.contains(pid))
            .map(|(pid, proc)| (*pid, proc.seq, proc.progressed))
            .collect();
        runner_ids.sort_by_key(|&(_, seq, _)| seq);
        runner_ids.truncate(RUNNER_DUMP_CAP);
        let mut proc_states = [0u32; 4];
        let mut admitted_procs = 0u32;
        for proc in inner.procs.values() {
            let slot = match proc.state {
                Residency::Resident => 0,
                Residency::Evicting => 1,
                Residency::Evicted => 2,
                Residency::Restoring => 3,
            };
            proc_states[slot] += 1;
            admitted_procs += u32::from(proc.admitted);
        }
        let accumulation = inner.accum.len() as u32;
        drop(inner);
        let (model, engine) = self.port.locus();
        let runners = runner_ids
            .into_iter()
            .map(|(pid, seq, progressed)| (seq, held_page_count(pid, model, engine), progressed))
            .collect();
        use Ordering::Relaxed;
        PlannerDiagnostics {
            device_pages_free,
            device_pages_total,
            host_slots_free,
            host_slots_total,
            disk_slots_free,
            disk_slots_total,
            rs_free,
            rs_total,
            accumulation,
            queue,
            proc_states,
            admitted_procs,
            runners,
            parks_total: self.stats.parks.load(Relaxed),
            serves_total: self.stats.serves.load(Relaxed),
            evictions_total: self.stats.evictions.load(Relaxed),
            eviction_deferrals_total: self.stats.eviction_deferrals.load(Relaxed),
            unmet_head_pages,
            unmet_head_kind,
            unmet_queued,
            bypassable_entries,
            bypassable_pages,
            eviction_rollbacks_total: self.stats.eviction_rollbacks.load(Relaxed),
            restores_total: self.stats.restores.load(Relaxed),
            restore_failures_total: self.stats.restore_failures.load(Relaxed),
            gate_parks_total: self.stats.gate_parks.load(Relaxed),
            cancelled_waits_total: self.stats.cancelled_waits.load(Relaxed),
            hog_failures_total: self.stats.hog_failures.load(Relaxed),
            starvations_total: self.stats.starvations.load(Relaxed),
            starvation_restarts_total: self.stats.starvation_restarts.load(Relaxed),
            salvages_total: self.stats.salvages.load(Relaxed),
            hoard_bypasses_total: self.stats.hoard_bypasses.load(Relaxed),
            e6_relaxations_total: self.stats.e6_relaxations.load(Relaxed),
            restore_absorb_short_total: self.stats.restore_absorb_short.load(Relaxed),
            host_swap_exhaustions_total: self.stats.host_swap_exhaustions.load(Relaxed),
            host_swap_unblocks_total: self.stats.host_swap_unblocks.load(Relaxed),
            d2h_pages_total: self.stats.d2h_pages.load(Relaxed),
            h2d_pages_total: self.stats.h2d_pages.load(Relaxed),
            disk_write_pages_total: self.stats.disk_write_pages.load(Relaxed),
            disk_read_pages_total: self.stats.disk_read_pages.load(Relaxed),
            disk_write_us_total: self.stats.disk_write_us.load(Relaxed),
            disk_read_us_total: self.stats.disk_read_us.load(Relaxed),
            d2h_rs_slots_total: self.stats.d2h_rs_slots.load(Relaxed),
            h2d_rs_slots_total: self.stats.h2d_rs_slots.load(Relaxed),
            d2h_copy_us_total: self.stats.d2h_copy_us.load(Relaxed),
            h2d_copy_us_total: self.stats.h2d_copy_us.load(Relaxed),
        }
    }

    pub fn debug_queue(&self) -> String {
        self.with_inner(|inner| {
            inner
                .queue
                .iter()
                .map(|(key, waiter)| {
                    let kind = match &waiter.kind {
                        WaitKind::Allocation {
                            demand,
                            outcome,
                            yielded,
                            ..
                        } => format!(
                            "alloc kv={:?} rs={} outcome={} yielded={yielded}",
                            demand.kv_pages,
                            demand.rs,
                            match outcome {
                                None => "none",
                                Some(Ok(_)) => "granted",
                                Some(Err(_)) => "err",
                            }
                        ),
                        WaitKind::Restore { demand } => {
                            format!("restore kv={} rs={}", demand.kv_pages, demand.rs)
                        }
                    };
                    format!("{key:?} pid={} {kind}", waiter.pid)
                })
                .collect::<Vec<_>>()
                .join("; ")
        })
    }

    pub fn kv_pressure_bucket(&self) -> u8 {
        let (swap_free, swap_total) = self.swap_stats();
        let ratio = |(free, total): (u32, u32)| {
            if total == 0 {
                0.0
            } else {
                f64::from(total.saturating_sub(free)) / f64::from(total)
            }
        };
        // Pages an idle process holds are headroom: the next demand evicts
        // them first. Counted as pressure, they would hold a launch at the
        // gateway that no fire is left to free them for. So are the pages
        // only an unused cache lease keeps (a finished process's indexed
        // prefix): the next reservation drops the lease, but a launch the
        // gateway holds back on their account never makes one.
        let idle: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.idle && proc.state == Residency::Resident)
                .map(|(pid, _)| *pid)
                .collect()
        });
        let (model, engine) = self.port.locus();
        let reclaimable: u32 = idle
            .into_iter()
            .map(|pid| held_page_count(pid, model, engine))
            .sum::<u32>()
            .saturating_add(self.port.reclaimable_idle());
        let (free, total) = self.port.device_stats();
        let device = ratio((free.saturating_add(reclaimable).min(total), total));
        let mut bucket = (device.max(ratio((swap_free, swap_total))) * 255.0).round() as u8;
        if self.nonresident.load(Ordering::Acquire) != 0
            || self.waiters.load(Ordering::Acquire) != 0
        {
            bucket = bucket.max(224);
        }
        bucket
    }
}
