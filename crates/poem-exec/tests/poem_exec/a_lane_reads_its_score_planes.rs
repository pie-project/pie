//! A lane reading `attn.scores` is told how the row factors: Qwen3-0.6B's
//! 16 query heads, each a run of 2048 keys (its 8 kv heads are not the
//! count), so an eviction program folds over every head.

use poem::Platform;
use poem_compiler::{Budget, Budgets, DeviceProfile, compile_axes};

#[test]
fn a_lane_reads_its_score_planes() {
    let row = poem_compiler::catalog::deployment("qwen3-0.6b-bf16-kv-bf16")
        .expect("the catalog lists Qwen3-0.6B");
    let trace = row.trace(Platform::Metal);
    let compiled = compile_axes(&trace, &Budgets::of(Budget::new(4, 256)), &DeviceProfile::default())
        .expect("the axes compile");
    let decl = poem_exec::feeds::class_outputs(&trace, &compiled)
        .expect("the outputs declare")
        .into_iter()
        .find(|out| out.decl.name == "attn.scores")
        .expect("a class reads out the scores")
        .decl;
    assert_eq!(decl.planes, vec![16, 2048], "{decl:?}");
    assert_eq!(decl.width, 16 * 2048);
    assert_eq!(decl.layers, 28, "a row a layer");
}
