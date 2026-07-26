//! The residency gate — the only guest-side vestige of KV contention under
//! Project Rainer (`rainer.md`).
//!
//! Every WIT host method that can touch pooled state passes through
//! [`residency_gate`] in its prologue. The fast path is one relaxed atomic
//! load (everyone resident — the overwhelmingly common case). When THIS
//! process has been evicted by the planner, the gate first settles the
//! process's own in-flight fire tail (releasing the fire leases the
//! eviction's quiescence wait needs), then parks until the planner restores
//! residency. There is no park protocol, no decline, no safe-point state —
//! the guest simply waits out its own eviction.

use anyhow::{Context, Result};

use crate::inferlet::ProcessCtx;

/// Fire-prologue residency gate: a no-op unless this process is out of the
/// resident set.
pub(crate) async fn residency_gate(ctx: &mut ProcessCtx) -> Result<()> {
    let Some(planner) = crate::planner::planner() else {
        return Ok(());
    };
    if planner.gate_open() {
        return Ok(());
    }
    let pid = ctx.id();
    if planner.is_resident(pid) {
        return Ok(()); // someone ELSE is evicted; we proceed
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
