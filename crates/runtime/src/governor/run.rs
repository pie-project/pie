//! The loop, on a pool: read the system, move the target.

use super::{Demand, Governor, Limits, Pressure, System, TICK, Tolerance};
use crate::store::extent::Extent;

/// A dedicated process's whole pool stands: there is nothing to answer.
pub fn start(extent: Extent, tolerance: Tolerance) {
    if tolerance == Tolerance::Dedicated {
        return;
    }
    crate::rt::spawn(watch(extent, tolerance, super::system()));
}

/// What one calm tick may add. A unified memory machine has no budget to
/// stop a burst but this: the system's complaint arrives after the damage.
const GROWTH_STEP: u64 = 512 * 1024 * 1024;

async fn watch(extent: Extent, tolerance: Tolerance, mut system: Box<dyn System>) {
    let mut governor = Governor::new(
        Limits {
            // One context's pages and its state, so one lane can always run.
            floor: extent.floor(),
            hard_limit: extent.hard_limit(),
            step: GROWTH_STEP,
            shrink_step: if extent.moves_by_copy() {
                u64::MAX
            } else {
                u64::from(extent.step()).saturating_mul(extent.page_bytes())
            },
        },
        tolerance,
        extent.held(),
    );
    let mut interval = crate::rt::time::interval(TICK);
    let mut last = None;
    let mut yielded = false;
    let mut idled = false;
    loop {
        let waiter = tokio::select! {
            _ = interval.tick() => false,
            () = extent.waiter() => true,
        };
        let asked = extent.asked();
        let demand = Demand {
            wanted: asked.wanted,
            live: asked.live,
        };
        let target = if waiter {
            governor.asked(demand)
        } else {
            let read = system.read();
            let refused = extent.take_refused();
            governor.tick(if refused { Pressure::Critical } else { read }, demand)
        };
        let split = extent.split(target, asked);
        if last != Some(split) {
            tracing::debug!(waiter, target, ?split, "pool target");
            extent.retarget(split);
            last = Some(split);
        }
        extent.set_can_rise(governor.can_rise());
        if governor.yielding() != yielded {
            yielded = governor.yielding();
            extent.yield_scratch(yielded).await;
        }
        if governor.idle() != idled {
            idled = governor.idle();
            if idled {
                extent.release_scratch().await;
            }
        }
    }
}
