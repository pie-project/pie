//! A stage's values by when the launches an engine runs first write and
//! last read them, and the scratch laid out of that.

use eta_compiler::codegen::launch::{LaunchRegion, LaunchStagePlan};
use eta_compiler::plan::{Dimension, RegionKind, is_row_vector, same_rows, value_rows};
use eta_ir::op::tags;
use eta_ir::types::MAX_RANK;

use super::{
    Area, ChunkLayout, Layout, Lifetime, Pieces, TooLarge, chunked_with, reusing_with,
};
use crate::extent::{Extents, Unresolvable, ValueDesc, describe, element_bytes};

/// Nodes of a stage an engine runs together. A node may lie in several
/// launches (a reduction whose partial pass and final pass are dispatched
/// apart): its result is written from the first and holds through the last,
/// and it reads its arguments through the last.
#[derive(Clone, Debug)]
pub struct Launch<'a> {
    /// Its nodes, in the order it runs them.
    pub nodes: &'a [u32],
    /// Each node is a step of its own and a value the launch both writes
    /// and reads holds to its last; otherwise the launch is one step.
    pub stepwise: bool,
    /// The blocks it runs across: past one, two of its values share bytes
    /// only as rows of one class.
    pub blocks: u32,
    /// The value whose rows its blocks walk.
    pub row_value: Option<u32>,
    pub row_alias: Option<u64>,
    /// Values that hold to its end whatever reads them.
    pub held: Vec<u32>,
    /// A temporary area of its own, alive over its steps alone, in place
    /// of the stage's ([`Launches::temporary`]): what its kernels keep
    /// while it runs, as the engine states it.
    pub keeps: Option<Keeps>,
}

/// What a launch keeps in a temporary area of its own: `bytes`, with
/// `reserve` bytes clear right below them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Keeps {
    pub reserve: u64,
    pub bytes: u64,
}

impl<'a> Launch<'a> {
    /// `region` as one launch across `blocks`: a generated region a step a
    /// node in the order its emitter runs them, a library region one step.
    #[must_use]
    pub fn of_region(region: &'a LaunchRegion, blocks: u32) -> Launch<'a> {
        Launch {
            nodes: if region.order.is_empty() {
                &region.nodes
            } else {
                &region.order
            },
            stepwise: region.kind == RegionKind::Generated,
            blocks,
            row_value: region.row_value,
            row_alias: region.row_alias,
            held: region
                .inputs
                .iter()
                .chain(region.outputs.iter())
                .copied()
                .chain(region.sinks.iter().map(|sink| sink.value))
                .collect(),
            keeps: None,
        }
    }

    /// `node` as a launch of its own, one step in one block, keeping
    /// `keeps` in an area of its own.
    #[must_use]
    pub fn of_node(node: &'a u32, keeps: Option<Keeps>) -> Launch<'a> {
        Launch {
            nodes: core::slice::from_ref(node),
            stepwise: false,
            blocks: 1,
            row_value: None,
            row_alias: None,
            held: Vec::new(),
            keeps,
        }
    }
}

/// How an engine runs a stage: what its scratch is laid out over.
#[derive(Clone, Debug, Default)]
pub struct Launches<'a> {
    /// The launches, in the order they run.
    pub each: Vec<Launch<'a>>,
    /// Nodes it launches nothing for. A result of one is written before
    /// the first launch; what one reads is `kept` or is not read.
    pub idle: Vec<u32>,
    /// Values written before the first launch, whatever launch also
    /// writes them.
    pub staged: Vec<u32>,
    /// Values read after the last launch.
    pub kept: Vec<u32>,
    /// Values that take no bytes of their own.
    pub hollow: Vec<u32>,
    /// Bytes held clear below the temporary area.
    pub reserve: u64,
    /// The bytes the stage's temporary area holds, alive across the stage:
    /// the most its launches' kernels keep there, those that keep an area
    /// of their own ([`Launch::keeps`]) aside.
    pub temporary: u64,
}

/// Each value's lifetime under `run`, or `None` where a node is neither
/// launched nor idle.
#[must_use]
pub fn value_lifetimes(plan: &LaunchStagePlan, run: &Launches<'_>) -> Option<Vec<Lifetime>> {
    let mut step_of = vec![u32::MAX; plan.ops.len()];
    let mut launch_of = vec![u32::MAX; plan.ops.len()];
    let mut first_step_of = vec![u32::MAX; plan.ops.len()];
    let mut first_launch_of = vec![u32::MAX; plan.ops.len()];
    let mut launch_first = Vec::with_capacity(run.each.len());
    let mut launch_last = Vec::with_capacity(run.each.len());
    let mut step = 0u32;
    for (index, launch) in run.each.iter().enumerate() {
        launch_first.push(step);
        for &node in launch.nodes {
            let index = u32::try_from(index).ok()?;
            let first = first_step_of.get_mut(node as usize)?;
            if *first == u32::MAX {
                *first = step;
                first_launch_of[node as usize] = index;
            }
            step_of[node as usize] = step;
            launch_of[node as usize] = index;
            if launch.stepwise {
                step = step.checked_add(1)?;
            }
        }
        if !launch.stepwise {
            step = step.checked_add(1)?;
        }
        launch_last.push(step.saturating_sub(1));
    }
    let end = step.saturating_sub(1);
    let mut idle = vec![false; plan.ops.len()];
    for &node in &run.idle {
        *idle.get_mut(node as usize)? = true;
    }
    if step_of
        .iter()
        .zip(&idle)
        .any(|(&step, &idle)| step == u32::MAX && !idle)
    {
        return None;
    }
    let launch_of_step = |step: u32| -> u32 {
        launch_first
            .partition_point(|&first| first <= step)
            .saturating_sub(1) as u32
    };
    let class_of = |launch: u32, value: u32| -> u64 {
        let Some(launch) = run.each.get(launch as usize) else {
            return Lifetime::SEQUENTIAL;
        };
        if launch.blocks <= 1 {
            return Lifetime::SEQUENTIAL;
        }
        let geometry = launch
            .row_value
            .and_then(|witness| plan.value_types.get(witness as usize))
            .and_then(|ty| value_rows(&ty.axes));
        let Some(geometry) = geometry else {
            return 0;
        };
        let Some(ty) = plan.value_types.get(value as usize) else {
            return 0;
        };
        let elem = element_bytes(ty.dtype);
        let alias = launch.row_alias;
        if ty.axes.len() >= 2
            && value_rows(&ty.axes).is_some_and(|shape| same_rows(shape, geometry, alias))
        {
            let width = match ty.axes.last() {
                Some(Dimension::Static(width)) => u64::from(*width),
                _ => return 0,
            };
            (1u64 << 40) | (width * elem)
        } else if is_row_vector(&ty.axes, geometry.0, geometry.1, alias) {
            (2u64 << 40) | elem
        } else {
            0
        }
    };
    let values = plan.value_types.len();
    let mut lifetimes = vec![
        Lifetime {
            dead: false,
            def: 0,
            last: 0,
            reusable: false,
            launch_def: 0,
            launch_last: 0,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        };
        values
    ];
    let mut alias_of = vec![u32::MAX; values];
    let bump = |lifetimes: &mut Vec<Lifetime>, alias_of: &[u32], mut value: u32, step: u32| {
        for _ in 0..=alias_of.len() {
            let Some(life) = lifetimes.get_mut(value as usize) else {
                return;
            };
            life.last = life.last.max(step);
            match alias_of.get(value as usize).copied() {
                Some(source) if source != u32::MAX => value = source,
                _ => return,
            }
        }
    };
    let mut result_base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        let step = step_of[node];
        let launched = step != u32::MAX;
        let written_in_full = !matches!(
            op.tag,
            tags::PIVOT_THRESHOLD | tags::CHAN_READ | tags::CHAN_TAKE
        );
        if launched {
            for result in 0..u32::from(op.result_count) {
                if let Some(life) = lifetimes.get_mut((result_base + result) as usize) {
                    life.def = first_step_of[node];
                    life.last = life.last.max(step);
                    life.reusable = written_in_full;
                    life.launch_def = first_launch_of[node];
                }
            }
        }
        if op.tag == tags::RESHAPE
            && let Some(&source) = op.args.first()
            && let Some(slot) = alias_of.get_mut(result_base as usize)
        {
            *slot = source;
        }
        result_base += u32::from(op.result_count);
        if !launched {
            continue;
        }
        let launch = launch_of[node];
        let stepwise = run
            .each
            .get(launch as usize)
            .is_some_and(|launch| launch.stepwise);
        for value in op.reads() {
            bump(&mut lifetimes, &alias_of, value, step);
            if stepwise
                && lifetimes
                    .get(value as usize)
                    .is_some_and(|life| life.launch_def == launch && life.last >= life.def)
                && let Some(&last) = launch_last.get(launch as usize)
            {
                bump(&mut lifetimes, &alias_of, value, last);
            }
        }
    }
    for (launch, &step) in run.each.iter().zip(&launch_last) {
        for &value in &launch.held {
            bump(&mut lifetimes, &alias_of, value, step);
        }
    }
    for &value in &run.staged {
        if let Some(life) = lifetimes.get_mut(value as usize) {
            life.def = 0;
            life.reusable = false;
            life.launch_def = 0;
        }
    }
    for &value in &run.kept {
        bump(&mut lifetimes, &alias_of, value, end);
    }
    for (value, life) in lifetimes.iter_mut().enumerate() {
        let value = value as u32;
        life.launch_last = launch_of_step(life.last);
        life.class_def = class_of(life.launch_def, value);
        life.class_last = class_of(life.launch_last, value);
        if life.class_last == 0
            && let Some(&last) = launch_last.get(life.launch_last as usize)
        {
            life.last = life.last.max(last);
        }
    }
    for &value in &run.hollow {
        if let Some(life) = lifetimes.get_mut(value as usize) {
            life.dead = true;
        }
    }
    Some(lifetimes)
}

/// The areas the launches of `run` that keep one take ([`Launch::keeps`]),
/// in launch order: each alive from its launch's first step to its last,
/// as a value its launch writes and reads, sharing no bytes with a value
/// its launch reads or writes, nor within a launch across blocks with a
/// row of any value.
#[must_use]
pub fn launch_areas(run: &Launches<'_>) -> Vec<Area> {
    let mut areas = Vec::new();
    let mut step = 0u32;
    for (index, launch) in run.each.iter().enumerate() {
        let first = step;
        let steps = if launch.stepwise {
            u32::try_from(launch.nodes.len()).unwrap_or(u32::MAX)
        } else {
            1
        };
        step = step.saturating_add(steps);
        let Some(keeps) = launch.keeps else {
            continue;
        };
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        let class = if launch.blocks > 1 {
            0
        } else {
            Lifetime::SEQUENTIAL
        };
        areas.push(Area {
            life: Lifetime {
                def: first,
                last: step.saturating_sub(1).max(first),
                dead: false,
                // Every kernel writes what it keeps there before it reads it.
                reusable: true,
                launch_def: index,
                launch_last: index,
                class_def: class,
                class_last: class,
            },
            reserve: keeps.reserve,
            bytes: keeps.bytes,
        });
    }
    areas
}

/// `plan`'s scratch under `run`: values, and the areas launches keep of
/// their own ([`launch_areas`]), share bytes where their lifetimes part;
/// each value its own, and each area its own past them, where a node is
/// neither launched nor idle.
pub fn lay_out(
    plan: &LaunchStagePlan,
    descriptors: &[ValueDesc],
    run: &Launches<'_>,
) -> Result<Layout, TooLarge> {
    let lifetimes = value_lifetimes(plan, run).unwrap_or_default();
    reusing_with(
        descriptors,
        &lifetimes,
        run.reserve,
        run.temporary,
        &launch_areas(run),
    )
}

/// `plan`'s scratch under `run` in `pieces` ([`layout_chunked`]): values
/// share bytes where their lifetimes part, each piece a run of adjacent
/// chunks inside a segment that holds the widest item whole, and each
/// value holds its own bytes where a node is neither launched nor idle.
pub fn lay_out_chunked(
    plan: &LaunchStagePlan,
    descriptors: &[ValueDesc],
    run: &Launches<'_>,
    pieces: Pieces,
) -> Result<ChunkLayout, TooLarge> {
    let lifetimes = value_lifetimes(plan, run).unwrap_or_default();
    chunked_with(
        descriptors,
        &lifetimes,
        run.reserve,
        run.temporary,
        &launch_areas(run),
        pieces,
    )
}

/// The rows each of `plan`'s regions walks: its row value's, one where it
/// names none.
#[must_use]
pub fn region_rows(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Vec<u32> {
    plan.fused
        .iter()
        .map(|region| {
            region
                .row_value
                .and_then(|value| descriptors.get(value as usize))
                .map_or(1, |descriptor| descriptor.rows.max(1))
        })
        .collect()
}

/// Which values are channel contents only regions `launches` refuses read:
/// nothing an engine launches carries them.
#[must_use]
pub fn read_only_unlaunched(
    plan: &LaunchStagePlan,
    launches: impl Fn(&LaunchRegion) -> bool,
) -> Vec<bool> {
    let values = plan.value_types.len();

    let mut producer = vec![usize::MAX; values];
    let mut result_base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        for result in 0..u32::from(op.result_count) {
            if let Some(slot) = producer.get_mut((result_base + result) as usize) {
                *slot = node;
            }
        }
        result_base += u32::from(op.result_count);
    }

    let mut skipped = vec![false; plan.ops.len()];
    let mut unread = vec![false; values];
    for region in plan.fused.iter().filter(|region| !launches(region)) {
        for &node in &region.nodes {
            if let Some(node) = skipped.get_mut(node as usize) {
                *node = true;
            }
        }
        for value in region.inputs.iter().copied().chain(
            region
                .nodes
                .iter()
                .filter_map(|&node| plan.ops.get(node as usize))
                .flat_map(|op| op.reads()),
        ) {
            if let Some(value) = unread.get_mut(value as usize) {
                *value = true;
            }
        }
    }

    for (node, op) in plan.ops.iter().enumerate() {
        if skipped.get(node).copied().unwrap_or(false) {
            continue;
        }
        for value in op.reads() {
            if let Some(value) = unread.get_mut(value as usize) {
                *value = false;
            }
        }
    }
    for region in plan.fused.iter().filter(|region| launches(region)) {
        for sink in &region.sinks {
            if let Some(value) = unread.get_mut(sink.value as usize) {
                *value = false;
            }
        }
    }

    for (value, drop) in unread.iter_mut().enumerate() {
        let tag = producer
            .get(value)
            .and_then(|&node| plan.ops.get(node))
            .map(|op| op.tag);
        if !matches!(tag, Some(tags::CHAN_READ | tags::CHAN_TAKE)) {
            *drop = false;
        }
    }
    unread
}

/// Every value of `plan` against `extents`, a value only unlaunched
/// regions read ([`read_only_unlaunched`]) described empty.
pub fn describe_launched(
    plan: &LaunchStagePlan,
    extents: &Extents,
    launches: impl Fn(&LaunchRegion) -> bool,
) -> Result<Vec<ValueDesc>, Unresolvable> {
    let empty = read_only_unlaunched(plan, launches);
    plan.value_types
        .iter()
        .enumerate()
        .map(|(value, declared)| {
            let described = describe(declared, extents)?;
            Ok(if empty.get(value).copied().unwrap_or(false) {
                ValueDesc {
                    len: 0,
                    rows: 0,
                    last: 0,
                    dims: [0; MAX_RANK],
                    ..described
                }
            } else {
                described
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scratch::{SELECT_RESERVE, layout};
    use eta_compiler::codegen::launch::{LaunchOp, LaunchPlanValue};
    use eta_ir::Dtype;

    const WIDE: u32 = 1024;

    fn op(tag: u8, result_count: u16, args: &[u32]) -> LaunchOp {
        LaunchOp {
            tag,
            result_count,
            args: args.to_vec(),
            channel: matches!(tag, tags::CHAN_TAKE | tags::CHAN_READ | tags::CHAN_PUT).then_some(0),
            ..Default::default()
        }
    }

    /// A stage of `ops`, every value a row of [`WIDE`] words.
    fn stage(ops: Vec<LaunchOp>) -> (LaunchStagePlan, Vec<ValueDesc>) {
        let values: usize = ops.iter().map(|op| usize::from(op.result_count)).sum();
        let plan = LaunchStagePlan {
            ops,
            value_types: vec![
                LaunchPlanValue {
                    dtype: Dtype::U32,
                    axes: vec![Dimension::Static(WIDE)],
                };
                values
            ],
            ..Default::default()
        };
        let descriptors = plan
            .value_types
            .iter()
            .map(|declared| describe(declared, &Extents::default()).expect("static"))
            .collect();
        (plan, descriptors)
    }

    /// Each of `launched` a launch of its own, the rest of `plan` idle.
    fn each_alone<'a>(plan: &LaunchStagePlan, launched: &'a [u32]) -> Launches<'a> {
        Launches {
            each: launched.iter().map(|node| Launch::of_node(node, None)).collect(),
            idle: (0..plan.ops.len() as u32)
                .filter(|node| !launched.contains(node))
                .collect(),
            ..Launches::default()
        }
    }

    #[test]
    fn lifetimes_every_case() {
        a_value_takes_the_bytes_of_one_an_earlier_launch_read_last();
        a_staged_value_is_written_before_the_first_launch();
        a_kept_value_holds_to_the_end_through_its_reshapes();
        a_hollow_value_takes_no_bytes();
        a_node_neither_launched_nor_idle_lays_each_value_its_own();
        the_reserve_and_the_temporary_bytes_reach_the_layout();
        a_launch_that_keeps_an_area_lays_it_over_its_own_steps();
    }

    /// take, a + a, b + b, c + c, put: one launch an op.
    fn chain() -> (LaunchStagePlan, Vec<ValueDesc>) {
        stage(vec![
            op(tags::CHAN_TAKE, 1, &[]),
            op(tags::ADD, 1, &[0, 0]),
            op(tags::ADD, 1, &[1, 1]),
            op(tags::ADD, 1, &[2, 2]),
            op(tags::CHAN_PUT, 0, &[3]),
        ])
    }

    fn a_value_takes_the_bytes_of_one_an_earlier_launch_read_last() {
        let (plan, descriptors) = chain();
        let launched = [1, 2, 3];
        let run = Launches {
            staged: vec![0],
            kept: vec![3],
            ..each_alone(&plan, &launched)
        };
        let lifetimes = value_lifetimes(&plan, &run).expect("every node is launched or idle");
        let spans: Vec<(u32, u32)> = lifetimes.iter().map(|life| (life.def, life.last)).collect();
        assert_eq!(spans, [(0, 0), (0, 1), (1, 2), (2, 2)]);
        assert!(!lifetimes[0].reusable, "the host writes a take's bytes");
        assert!(lifetimes[1..].iter().all(|life| life.reusable));

        let laid = lay_out(&plan, &descriptors, &run).expect("fits");
        assert_eq!(
            laid.values[2], laid.values[0],
            "the take was last read a launch before"
        );
        assert_ne!(laid.values[1], laid.values[0], "read where it is written");
        assert_ne!(laid.values[3], laid.values[2]);
        assert_eq!(laid.values[3], laid.values[1]);
        assert!(laid.total < layout(&descriptors).expect("fits").total);
    }

    fn a_staged_value_is_written_before_the_first_launch() {
        // take, a + a, a constant, b + c, put: the constant is launched
        // third and the host writes it before the first.
        let (plan, descriptors) = stage(vec![
            op(tags::CHAN_TAKE, 1, &[]),
            op(tags::ADD, 1, &[0, 0]),
            op(tags::CONST, 1, &[]),
            op(tags::ADD, 1, &[1, 2]),
            op(tags::CHAN_PUT, 0, &[3]),
        ]);
        let launched = [1, 2, 3];
        let launched_only = Launches {
            kept: vec![3],
            ..each_alone(&plan, &launched)
        };
        let lifetimes = value_lifetimes(&plan, &launched_only).expect("covered");
        assert_eq!((lifetimes[2].def, lifetimes[2].reusable), (1, true));
        let laid = lay_out(&plan, &descriptors, &launched_only).expect("fits");
        assert_eq!(laid.values[2], laid.values[0], "it would take the take's");

        let staged = Launches {
            staged: vec![0, 2],
            ..launched_only
        };
        let lifetimes = value_lifetimes(&plan, &staged).expect("covered");
        assert_eq!((lifetimes[2].def, lifetimes[2].reusable), (0, false));
        let laid = lay_out(&plan, &descriptors, &staged).expect("fits");
        assert_ne!(laid.values[2], laid.values[0]);
        assert_ne!(laid.values[2], laid.values[1]);
    }

    fn a_kept_value_holds_to_the_end_through_its_reshapes() {
        // take, a + a, its reshape (put), c + c, d + d (put): the reshape
        // is not launched, and its source holds what the put reads.
        let (plan, descriptors) = stage(vec![
            op(tags::CHAN_TAKE, 1, &[]),
            op(tags::ADD, 1, &[0, 0]),
            op(tags::RESHAPE, 1, &[1]),
            op(tags::ADD, 1, &[1, 1]),
            op(tags::ADD, 1, &[3, 3]),
            op(tags::CHAN_PUT, 0, &[2]),
            op(tags::CHAN_PUT, 0, &[4]),
        ]);
        let launched = [1, 3, 4];
        let unkept = Launches {
            staged: vec![0],
            hollow: vec![2],
            ..each_alone(&plan, &launched)
        };
        let laid = lay_out(&plan, &descriptors, &unkept).expect("fits");
        assert_eq!(
            laid.values[4], laid.values[1],
            "nothing stated reads the sum after the launch that last does"
        );

        let kept = Launches {
            kept: vec![2, 4],
            ..unkept
        };
        let lifetimes = value_lifetimes(&plan, &kept).expect("covered");
        assert_eq!(lifetimes[1].last, 2, "the last launch's step");
        assert_eq!(lifetimes[4].last, 2);
        let laid = lay_out(&plan, &descriptors, &kept).expect("fits");
        assert_ne!(laid.values[4], laid.values[1]);
        assert_ne!(laid.values[3], laid.values[1]);
    }

    fn a_hollow_value_takes_no_bytes() {
        let (plan, descriptors) = chain();
        let launched = [1, 2, 3];
        let run = Launches {
            staged: vec![0],
            kept: vec![3],
            hollow: vec![1],
            ..each_alone(&plan, &launched)
        };
        let lifetimes = value_lifetimes(&plan, &run).expect("covered");
        assert!(lifetimes[1].dead);
        let laid = lay_out(&plan, &descriptors, &run).expect("fits");
        assert_eq!(laid.values[1], 0);
        let whole = lay_out(
            &plan,
            &descriptors,
            &Launches {
                hollow: Vec::new(),
                ..run
            },
        )
        .expect("fits");
        assert!(laid.total <= whole.total);
    }

    fn a_node_neither_launched_nor_idle_lays_each_value_its_own() {
        let (plan, descriptors) = chain();
        let launched = [1, 2];
        let run = Launches {
            idle: vec![0, 4],
            ..each_alone(&plan, &launched)
        };
        assert_eq!(value_lifetimes(&plan, &run), None);
        assert_eq!(
            lay_out(&plan, &descriptors, &run).expect("fits"),
            layout(&descriptors).expect("fits")
        );
    }

    fn the_reserve_and_the_temporary_bytes_reach_the_layout() {
        let (plan, descriptors) = chain();
        let launched = [1, 2, 3];
        let bare = Launches {
            staged: vec![0],
            kept: vec![3],
            ..each_alone(&plan, &launched)
        };
        let laid = lay_out(&plan, &descriptors, &bare).expect("fits");
        let stated = 1 << 20;
        let held = lay_out(
            &plan,
            &descriptors,
            &Launches {
                reserve: SELECT_RESERVE,
                temporary: stated,
                ..bare
            },
        )
        .expect("fits");
        assert_eq!(held.values, laid.values);
        assert!(held.temporary >= laid.temporary + SELECT_RESERVE);
        assert_eq!(laid.temporary_bytes, 0);
        assert_eq!(held.temporary_bytes, stated);
    }

    /// A launch that keeps an area lays it alive over its own steps, as a
    /// value it writes and reads: over the chain, the last add's area
    /// takes the bytes of the take's value and the first add's result,
    /// dead by then, and none of what that add reads or writes; a launch
    /// across blocks shares its area with no row of its own.
    fn a_launch_that_keeps_an_area_lays_it_over_its_own_steps() {
        let (plan, descriptors) = chain();
        let launched = [1u32, 2, 3];
        let keeps = Keeps {
            reserve: SELECT_RESERVE,
            bytes: 2 * u64::from(WIDE) * 4,
        };
        let run = Launches {
            each: launched
                .iter()
                .map(|node| Launch::of_node(node, (*node == 3).then_some(keeps)))
                .collect(),
            idle: vec![0, 4],
            staged: vec![0],
            kept: vec![3],
            ..Launches::default()
        };
        let [area] = &launch_areas(&run)[..] else {
            panic!("the last add alone keeps an area");
        };
        assert_eq!((area.life.def, area.life.last, area.life.launch_def), (2, 2, 2));
        assert_eq!((area.reserve, area.bytes), (keeps.reserve, keeps.bytes));
        let laid = lay_out(&plan, &descriptors, &run).expect("fits");
        let [first] = laid.areas[..] else {
            panic!("one area laid");
        };
        let (low, high) = (first - keeps.reserve, first + keeps.bytes);
        let word = u64::from(WIDE) * 4;
        for value in [2usize, 3] {
            let at = laid.values[value];
            assert!(at + word <= low || high <= at, "value {value}: {laid:?}");
        }
        let take = laid.values[0];
        assert!(low <= take + word && take < high, "{laid:?}");
        assert_eq!(laid.temporary_bytes, 0);

        let mut wide = Launches {
            each: run.each.clone(),
            ..Launches::default()
        };
        wide.each[2].blocks = 4;
        let [across] = &launch_areas(&wide)[..] else {
            panic!("one area");
        };
        assert_eq!((across.life.class_def, across.life.class_last), (0, 0));
    }
}
