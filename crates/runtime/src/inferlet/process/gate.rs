use anyhow::{Context, Result};

use crate::inferlet::ProcessCtx;

pub(crate) async fn residency_gate(ctx: &mut ProcessCtx) -> Result<()> {
    if ctx.is_resident_fast() {
        return Ok(());
    }
    let pid = ctx.id();
    let Some(planner) = crate::planner::planner_of(pid) else {
        return Ok(());
    };
    if planner.is_resident(pid) {
        return Ok(());
    }
    drain_pending_fires(ctx).await?;
    planner
        .wait_resident(pid)
        .await
        .context("wait for KV residency")?;
    Ok(())
}

pub(crate) async fn drain_pending_fires(ctx: &mut ProcessCtx) -> Result<()> {
    let pipelines = ctx.residency_pipelines();
    for fires in pipelines {
        crate::pipeline::fire::finalize_all(ctx, &fires, false).await?;
    }
    Ok(())
}
