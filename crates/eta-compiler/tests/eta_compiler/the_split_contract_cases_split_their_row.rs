//! The contract's split cases (`contract::split_stage`): a one-row region
//! that reads the model's row through views runs across its lane's blocks.

use eta_compiler::plan::compile_bound;
use eta_ir::op::tags;

/// The shared split cases (`contract::split_stage`) split their row, and the
/// interpreter answers them.
#[test]
fn the_split_contract_cases_split_their_row() {
    for case in eta_compiler::contract::split_stage() {
        let stages = compile_bound(&case.bound);
        let package = eta_compiler::codegen::launch::build(&case.bound, &stages);
        let plan = package.plans.first().expect("one plan");
        assert!(
            plan.fused.iter().any(|region| region.splits_rows
                && region
                    .nodes
                    .iter()
                    .any(|&node| plan.ops[node as usize].tag == tags::OUTPUT)),
            "{}: its row runs on one block",
            case.name
        );
        let want = case.expected();
        assert_eq!(want.len(), 2, "{}", case.name);
    }
}
