//! The temporal plan: a trace run as launches over time on one device —
//! regions of nodes, how each lowers (always, captured, switched), the
//! streams they fork onto, the order the request classes' graphs seriate in
//! and the arena they carve. [`crate::compile_axes`] is [`crate::analysis`]
//! then [`plan`].

use poem_ir::{ClassTable, RowAxis, Trace};

use crate::analysis::Analysis;
use crate::{
    AxisPlan, Budgets, CompiledModel, Concurrency, DeviceProfile, Error, Region, arena, layout,
    lowering, region, stream, unit,
};

/// The temporal plan of `trace`, analysed as `analysis`, under `budgets` on
/// `profile`'s device.
pub fn plan(
    trace: &Trace,
    analysis: Analysis,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Result<CompiledModel, Error> {
    let budget = &budgets.tokens;
    let Analysis {
        classes, expert, ..
    } = analysis;
    // Scheduled by dependencies (`compiler.dag_streams`): each op's lane is
    // found over one-op regions, then regions break only where lanes change.
    let lanes = if profile.dag_streams {
        let every: std::collections::BTreeSet<u32> = (0..trace.nodes.len() as u32).collect();
        let mut fine = region::coalesce_breaking(trace, &classes, &every)?;
        region::hoist(trace, &mut fine)?;
        let (_, fine_units_of) = unit::partition(&fine)?;
        lowering::lower(trace, &mut fine, &classes, budget, profile);
        Some(stream::dag_lanes(trace, &mut fine, &fine_units_of, profile))
    } else {
        None
    };
    let breaks: std::collections::BTreeSet<u32> = match &lanes {
        Some((_, breaks)) => breaks.clone(),
        None => std::collections::BTreeSet::new(),
    };
    let mut regions = region::coalesce_breaking(trace, &classes, &breaks)?;

    region::hoist(trace, &mut regions)?;

    let (units, units_of) = unit::partition(&regions)?;

    lowering::lower(trace, &mut regions, &classes, budget, profile);

    let streams = if let Some((lanes, _)) = &lanes {
        stream::fork_dag(trace, &mut regions, &units_of, profile, Some(lanes))
    } else {
        stream::fork_with(trace, &mut regions, &units_of, profile)
    };
    let concurrency =
        Concurrency::with_pairs(&regions, trace.nodes.len(), streams.pairs.iter().copied());

    let (order, fallback) = layout::seriate(
        trace,
        &regions_on(&regions, &units_of, &units, RowAxis::Tokens),
        &classes,
        &budget.buckets,
        budget.max_tokens,
        profile,
    );
    let patches = axis_plan(
        RowAxis::Patches,
        trace,
        &regions,
        &units_of,
        &units,
        &classes,
        budgets,
        profile,
    );
    let voxels = axis_plan(
        RowAxis::Voxels,
        trace,
        &regions,
        &units_of,
        &units,
        &classes,
        budgets,
        profile,
    );

    // Side streams only up to `forked_until` rows: the arena in order for
    // every fire, and one under the streams' concurrency at those rows' budgets
    // for the fires that fork.
    let (arena, arena_forked) = if lanes.is_some() && streams.forked_until > 0 {
        let until = streams.forked_until;
        let serial = Concurrency::sequential(&regions, trace.nodes.len());
        let main = arena::carve(trace, budgets, &classes, &serial)?;
        let mut narrow = budgets.clone();
        narrow.tokens.max_tokens = narrow.tokens.max_tokens.min(until);
        narrow.tokens.max_lanes = narrow.tokens.max_lanes.min(until);
        narrow.tokens.buckets.retain(|&b| b <= until);
        let forked = arena::carve(trace, &narrow, &classes, &concurrency)?;
        (main, Some(forked))
    } else {
        (arena::carve(trace, budgets, &classes, &concurrency)?, None)
    };

    let voxel_growth = arena::voxel_growth(trace, &classes);
    Ok(CompiledModel {
        voxel_growth,
        classes,
        regions,
        order,
        fallback,
        arena,
        arena_forked,
        concurrency,
        streams,
        fold_refused: unit::fold_refused(&units),
        expert,
        units,
        units_of,
        patches,
        voxels,
    })
}

fn regions_on(
    regions: &[Region],
    units_of: &[u32],
    units: &[RowAxis],
    axis: RowAxis,
) -> Vec<Region> {
    let Some(unit) = units.iter().position(|held| *held == axis) else {
        return Vec::new();
    };
    let unit = unit as u32;
    regions
        .iter()
        .enumerate()
        .filter(|(r, _)| units_of.get(*r).copied().unwrap_or(0) == unit)
        .map(|(_, region)| region.clone())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn axis_plan(
    axis: RowAxis,
    trace: &Trace,
    regions: &[Region],
    units_of: &[u32],
    units: &[RowAxis],
    classes: &ClassTable,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Option<AxisPlan> {
    let ladder = budgets.ladder(axis)?;
    if !units.contains(&axis) {
        return None;
    }
    let on_axis = regions_on(regions, units_of, units, axis);
    let (order, fallback) = layout::seriate(
        trace,
        &on_axis,
        classes,
        ladder.buckets,
        ladder.max_rows,
        profile,
    );
    Some(AxisPlan {
        axis,
        order,
        fallback,
    })
}
