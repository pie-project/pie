use eta_ir::op::Op;

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;

const V: u32 = 8;

#[test]
fn the_name_table_is_sorted_and_indices_are_remapped() {
    let mask = Channel::seeded([4], dtype::u32);
    let out = Channel::new([1], dtype::i32);
    let mut b = Builder::new(V, 2);
    b.outputs(crate::outputs::logits(V));
    b.stage(Stage::Epilogue, move || {
        out.put(cast(
            reduce_argmax(intrinsics::output(0, "logits")),
            dtype::i32,
        ));
    });
    b.stage(Stage::Prologue, move || {
        intrinsics::kernel::attn_page_mask(mask.read());
    });
    let t = b.build().expect("an output and a sink trace");
    let c = t.container();
    assert_eq!(
        c.names,
        vec!["attn_page_mask".to_string(), "logits".to_string()],
        "the name table is sorted, not in first-use order"
    );
    let named = |stage: Stage| {
        c.stages
            .iter()
            .find(|s| s.stage == stage)
            .expect("the stage")
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Output { index, .. } | Op::SinkCall { name: index, .. } => Some(*index),
                _ => None,
            })
            .expect("a named op")
    };
    assert_eq!(c.names[named(Stage::Epilogue) as usize], "logits");
    assert_eq!(c.names[named(Stage::Prologue) as usize], "attn_page_mask");
}
