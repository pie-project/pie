use std::io::Read;

use model_ir::plan::Plan;

fn main() {
    let mut json = String::new();
    std::io::stdin().read_to_string(&mut json).unwrap();
    let plan: Plan = serde_json::from_str(&json).expect("stdin is not a plan");
    let lowered = model_compiler::sweep::lower(&plan);

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
    for (i, lane) in lowered.lanes.iter().enumerate() {
        println!("  lane {i}: words {:?}, {} ops", lane.words, lane.ops.len());
    }
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
