use std::io::Read;

use model_compiler::program::{Program, Slot};
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
                report(program);
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

fn report(program: &Program) {
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
    println!(
        "    {} steps, {arena} arena slots, {aliases} aliases, row_pitch {} bytes",
        program.steps.len(),
        program.row_pitch
    );
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
}
