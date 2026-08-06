//! The residency gate — the only guest-side vestige of KV contention under
//! Project Rainer (`rainer.md`).
//!
//! Every WIT host method that can touch pooled state passes through
//! [`residency_gate`] in its prologue. The fast path is one relaxed atomic
//! load of THIS process's own residency flag. When this process has been
//! evicted by the planner, the gate first settles the process's own
//! in-flight fire tail (releasing the fire leases the eviction's quiescence
//! wait needs), then parks until the planner restores residency. There is
//! no park protocol, no decline, no safe-point state — the guest simply
//! waits out its own eviction.
//!
//! The fast path used to be the FLEET-wide `Planner::gate_open()`
//! ("nobody at all is evicted"), falling through to `Planner::is_resident`
//! — which takes the planner mutex — whenever it was false. That inverted
//! the cost exactly where it hurt: an idle fleet paid nothing, and a
//! contended fleet, where some process is evicted essentially always, paid
//! a global-lock acquisition on EVERY host call. Reading a per-process
//! flag is unconditional and costs the same either way.

use anyhow::{Context, Result};

use crate::inferlet::ProcessCtx;

/// Fire-prologue residency gate: a no-op unless this process is out of the
/// resident set.
pub(crate) async fn residency_gate(ctx: &mut ProcessCtx) -> Result<()> {
    if ctx.is_resident_fast() {
        return Ok(());
    }
    let Some(planner) = crate::planner::planner() else {
        return Ok(());
    };
    let pid = ctx.id();
    // Re-check under the lock: the mirror is refreshed at planner-lock
    // release, so a stale `false` here only costs this one confirmation.
    if planner.is_resident(pid) {
        return Ok(());
    }
    // Settle our own submitted tail: the parked task must hold no pins, and
    // these finalizations release the fire leases the eviction quiesces on.
    // `wait_resident` re-posts the process-wide leave before parking (the
    // lane-resurrection seal wedge, CONTENTION_FOLLOWUP.md §15.2).
    drain_pending_fires(ctx).await?;
    planner
        .wait_resident(pid)
        .await
        .context("wait for KV residency")?;
    Ok(())
}

/// Finalize every pending pipeline op of this process, in submit order.
pub(crate) async fn drain_pending_fires(ctx: &mut ProcessCtx) -> Result<()> {
    let pipelines = ctx.residency_pipelines();
    for fires in pipelines {
        crate::pipeline::fire::finalize_all(ctx, &fires, false).await?;
    }
    Ok(())
}
