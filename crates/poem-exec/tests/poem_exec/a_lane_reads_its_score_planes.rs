//! A lane reading the scores is told each row as the load lays it: a mass a
//! key of the load's context (`attn.scores`, 2048 keys) and a mass a kv page
//! of it (`attn.scores.pages`, 128 pages of 16), once, not a layer at a time.

use poem::Platform;
use poem_compiler::{Budget, Budgets, DeviceProfile, compile_axes};

#[test]
fn a_lane_reads_its_score_planes() {
    let row = poem_compiler::catalog::deployment("qwen3-0.6b-bf16-kv-bf16")
        .expect("the catalog lists Qwen3-0.6B");
    let trace = row.trace(Platform::Metal);
    let compiled = compile_axes(
        &trace,
        &Budgets::of(Budget::new(4, 256)),
        &DeviceProfile::default(),
    )
    .expect("the axes compile");
    let outputs =
        poem_exec::feeds::class_outputs(&trace, &compiled, 2048, 16).expect("the outputs declare");
    let decl = |name: &str| {
        outputs
            .iter()
            .find(|out| out.decl.name == name)
            .unwrap_or_else(|| panic!("a class reads out `{name}`"))
            .decl
            .clone()
    };
    let keys = decl("attn.scores");
    assert_eq!(
        (keys.planes.clone(), keys.width, keys.layers),
        (vec![1, 2048], 2048, 0),
        "{keys:?}"
    );
    let pages = decl("attn.scores.pages");
    assert_eq!(
        (pages.planes.clone(), pages.width, pages.layers),
        (vec![1, 128], 128, 0),
        "{pages:?}"
    );
}
