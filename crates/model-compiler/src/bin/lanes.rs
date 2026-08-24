use std::io::Read;

use model_compiler::program::{Call, Program, Rows, Slot, Why};
use model_ir::plan::Plan;

fn main() {
    let mut json = String::new();
    std::io::stdin().read_to_string(&mut json).unwrap();
    let plan: Plan = serde_json::from_str(&json).expect("stdin is not a plan");
    let lanes = model_compiler::sweep::lanes(&plan);
    let bound = model_compiler::program::bound(&plan);

    println!("plan `{}` on {:?}", plan.name, plan.plane);
    println!(
        "  {} facts {:?}, {} params, {} caches, {} ops, {} seams",
        plan.facts.len(),
        plan.facts,
        plan.params.len(),
        plan.caches.len(),
        plan.ops.len(),
        plan.seams.len()
    );
    let (mut built, mut refused) = (0, 0);
    for (i, lane) in lanes.iter().enumerate() {
        println!("  lane {i}: words {:?}, {} ops", lane.words, lane.ops.len());
        match &bound[i] {
            Ok(program) => {
                built += 1;
                report(&plan, program);
            }
            Err(refusal) => {
                refused += 1;
                for gap in &refusal.gaps {
                    println!("    REFUSED: {gap}");
                }
            }
        }
    }
    println!("  programs: {built} built, {refused} refused");

    resolution(&plan);
}

/// EVERY DISTINCT KERNEL THE PLAN STATES, joined against the plane, in the
/// order the plan first states each.
///
/// This is what `sweep::resolve` handed back as a `Resolution` — the second
/// derivation of `program::call_for`'s three-way answer, and the only reader
/// either of them had. The join is one call now; the three lists are this
/// binary's own formatting, which is where a report's shape belongs.
fn resolution(plan: &Plan) {
    let (mut resolved, mut unclaimed, mut wrong_plane) = (Vec::new(), Vec::new(), Vec::new());
    let mut seen: Vec<&str> = Vec::new();
    for op in &plan.ops {
        let kernel = op.kernel.as_str();
        if seen.contains(&kernel) {
            continue;
        }
        seen.push(kernel);
        match model_compiler::program::call_for(plan.plane, kernel) {
            // A tier-2 call is answered `tier2::` and a claimed point
            // `points::`, because the two reach a driver by different doors:
            // one through the generated dispatch, the other through a
            // staging shim. A `canon` row already spells its own symbol.
            Ok(Call::Tier2(point)) => resolved.push((kernel, format!("tier2::{point}"))),
            Ok(Call::Point(point)) => resolved.push((kernel, format!("points::{point}"))),
            Ok(Call::Symbol(symbol)) => resolved.push((kernel, symbol.to_string())),
            Err(Why::Unclaimed) => unclaimed.push(kernel),
            Err(Why::WrongPlane) => wrong_plane.push(kernel),
            // Never answered by a join over a NAME: it takes a bound
            // statement, and the lane reports above are where it shows up.
            Err(Why::Unsized) => unreachable!("`call_for` answers no width"),
        }
    }
    println!(
        "  resolution: {} resolved, {} unresolved, {} violations",
        resolved.len(),
        unclaimed.len(),
        wrong_plane.len()
    );
    for (role, symbol) in &resolved {
        println!("    {role} -> {symbol}");
    }
    for role in &unclaimed {
        println!("    {role} -> UNCLAIMED");
    }
    for role in &wrong_plane {
        println!("    {role} -> WRONG PLANE");
    }
}

fn report(plan: &Plan, program: &Program) {
    let arena = program
        .slots
        .iter()
        .filter(|s| matches!(s, Slot::Arena { .. }))
        .count();
    let aliases = program
        .slots
        .iter()
        .filter(|s| matches!(s, Slot::Alias(_)))
        .count();
    // THE BOUND BESIDE THE PITCH. `live_bound` is the arena's busiest
    // instant — the floor no layout can beat — so a pitch printed without it
    // says how big the arena is and not whether it is as small as it can be,
    // which is the only interesting half. `clashes` prints only when it has
    // something to say, which is never on a program the walk built.
    let bound = model_compiler::program::live_bound(plan, program);
    println!(
        "    {} steps, {arena} arena slots, {aliases} aliases, row_pitch {} bytes (bound {bound})",
        program.steps.len(),
        program.row_pitch
    );
    let clashes = model_compiler::program::clashes(plan, program);
    if !clashes.is_empty() {
        println!("    CLASHES: {} live pairs share bytes", clashes.len());
    }
    let widest = program
        .slots
        .iter()
        .filter_map(|s| match s {
            Slot::Arena { width, dtype, .. } => Some((*width, *dtype)),
            _ => None,
        })
        .max_by_key(|(width, _)| *width);
    if let Some((width, dtype)) = widest {
        println!("    widest slot: {width} x {dtype:?}");
    }
    // ONLY WHEN THERE ARE ANY. A plan with no router mints no routed slot,
    // and a report that printed "0 routed" on every dense lane would move
    // every dense SKU's output for a number that is always the same.
    let mut routed: Vec<u32> = program
        .slots
        .iter()
        .filter_map(|s| match s {
            Slot::Arena {
                rows: Rows::FireTimes(k),
                ..
            } => Some(*k),
            _ => None,
        })
        .collect();
    if !routed.is_empty() {
        let slots = routed.len();
        routed.sort_unstable();
        routed.dedup();
        println!("    {slots} routed slots at rows x {routed:?}");
    }
}
