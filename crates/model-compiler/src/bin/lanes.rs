use std::io::Read;

use model_compiler::program::{Program, Rows, Slot};
use model_ir::plan::Plan;

fn main() {
    let mut json = String::new();
    std::io::stdin().read_to_string(&mut json).unwrap();
    let plan: Plan = serde_json::from_str(&json).expect("stdin is not a plan");
    let lowered = model_compiler::sweep::lower(&plan);
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
    for (i, lane) in lowered.lanes.iter().enumerate() {
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

    let r = &lowered.resolution;
    println!(
        "  resolution: {} resolved, {} unresolved, {} violations",
        r.resolved.len(),
        r.unresolved.len(),
        r.violations.len()
    );
    for (role, symbol) in &r.resolved {
        println!("    {role} -> {symbol}");
    }
    for role in &r.unresolved {
        println!("    {role} -> UNCLAIMED");
    }
    for v in &r.violations {
        println!("    {v} -> WRONG PLANE");
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
