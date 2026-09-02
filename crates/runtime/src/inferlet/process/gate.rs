//! The residency gate. Every WIT host method that can touch pooled state
//! passes through [`residency_gate`] in its prologue: a relaxed atomic load
//! on the fast path, or, if evicted by the planner, settling this process's
//! in-flight fire tail and parking until residency is restored.

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
    // The parked task must hold no pins; these finalizations release the
    // fire leases the eviction quiesces on.
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
