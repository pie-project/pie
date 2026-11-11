//! A process's fires on their way through the engine: while one is, the
//! process is not stalled, whatever else of it is parked.
//!
//! A process whose lanes ask apart (text-completion-bench's batch) parked one
//! lane while the others' fires ran; read as stalled, the fleet looked
//! wedged and the process was reclaimed and replayed, often thousands of
//! times, before the ask that could never fit parked and failed. A fire
//! settles once the engine runs it, and its settling pokes the planner, so a
//! parked lane waits for it instead.
//!
//! What a fire waits on before it runs is never the parked ask of its own
//! process: the ask is quoted before the fire's slots are bound, a group
//! fire (contract 15) asks once for all its lanes, and a lane that parks
//! leaves its frame quorum first (`notify_lane_close`: its partial frame is
//! cut to the fires it has, and it is no longer awaited), so no frame, its
//! own or its peers', waits for it. A guest's own wait on its parked lane (a lockstep
//! barrier) holds no fire: the fires it has submitted run and settle all the
//! same. A fire can still wait on its guest (a host-written input), and that
//! guest on the parked lane; so a fire counts as progress only for
//! [`FIRE_PROGRESS`] since the process last began or settled one, past which
//! it is stalled as before.

use std::sync::{Arc, Weak};
use std::time::Duration;

use super::{ProcessId, ResidencyPlanner};

/// How long a process's fires in flight keep it from being stalled with
/// none of them settling.
pub(crate) const FIRE_PROGRESS: Duration = crate::pipeline::fire::SEAT_WAIT;

/// One fire of a process on its way through the engine, from its slots'
/// binding until the engine settles it (or, never submitted, until dropped).
pub struct FireInFlight {
    planner: Weak<ResidencyPlanner>,
    pid: ProcessId,
    counted: bool,
}

impl ResidencyPlanner {
    /// `pid` binds a fire's slots: it counts as in flight until the token
    /// drops.
    pub fn fire_in_flight(self: &Arc<Self>, pid: ProcessId) -> FireInFlight {
        let counted = self.with_inner(|inner| {
            inner.procs.get_mut(&pid).is_some_and(|proc| {
                proc.fires += 1;
                proc.fired_at = crate::rt::Instant::now();
                proc.fire_lapsed = false;
                true
            })
        });
        FireInFlight {
            planner: Arc::downgrade(self),
            pid,
            counted,
        }
    }

    fn fire_landed(self: &Arc<Self>, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.fires = proc.fires.saturating_sub(1);
                proc.fired_at = crate::rt::Instant::now();
                proc.fire_lapsed = false;
            }
        });
        // After the count drops: a poke read before it would see the fire
        // still in flight and wait on a settling that has passed.
        self.pages_freed();
    }

    /// A parked process whose fires have stood unsettled past
    /// [`FIRE_PROGRESS`] is stalled now: told once, so the planner looks
    /// again. The parked waiters call this as they wait.
    pub(super) fn note_lapsed_fires(self: &Arc<Self>) {
        let lapsed = self.with_inner(|inner| {
            let mut lapsed = false;
            for proc in inner.procs.values_mut() {
                if proc.fires > 0 && !proc.fire_lapsed && proc.fired_at.elapsed() >= FIRE_PROGRESS {
                    proc.fire_lapsed = true;
                    lapsed = true;
                }
            }
            lapsed
        });
        if lapsed {
            self.poke();
        }
    }
}

impl Drop for FireInFlight {
    fn drop(&mut self) {
        if self.counted
            && let Some(planner) = self.planner.upgrade()
        {
            planner.fire_landed(self.pid);
        }
    }
}
