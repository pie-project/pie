use poem::fact;
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, RaggedMask, Request, Stream, Value, Weight,
    ops, seam, trace_hybrid,
};
use poem_ir::{Layout, Operation};

const WIDTH: u32 = 32;
const HEAD_DIM: u32 = 8;

struct LastJointBlock;

impl ForwardHybrid for LastJointBlock {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let (txt, img) = (
            inputs.on(fact::stream(Stream::Text)),
            inputs.on(!fact::stream(Stream::Text)),
        );
        let w = |name: &str| Weight::sym(name, [u64::from(WIDTH), u64::from(WIDTH)], Dtype::Bf16);
        let x_txt = txt.latents(0, WIDTH, Dtype::Bf16);
        let x_img = img.latents(1, WIDTH, Dtype::Bf16);
        let project = |x: &Value, prefix: &str| {
            (
                ops::linear::matmul(x, &w(&format!("{prefix}.q"))),
                ops::linear::matmul(x, &w(&format!("{prefix}.k"))),
                ops::linear::matmul(x, &w(&format!("{prefix}.v"))),
            )
        };
        let (qt, kt, vt) = project(&x_txt, "txt");
        let (qi, ki, vi) = project(&x_img, "img");
        let perm = inputs.row_permutation();
        let indptr = inputs.group_indptr();
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&Value::merge(vec![qt, qi]), &perm),
            &ops::layout::pack_rows(&Value::merge(vec![kt, ki]), &perm),
            &ops::layout::pack_rows(&Value::merge(vec![vt, vi]), &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            0.35,
            RaggedMask::GroupBlockDiagonal,
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        let o_img = o.on(!fact::stream(Stream::Text));
        let out = ops::linear::matmul(&o_img, &w("img.o"));
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

#[test]
fn the_pack_and_unpack_of_a_head_only_block_are_demanded_in_both_streams() {
    let trace = trace_hybrid("last_joint", &LastJointBlock, Platform::Cuda);
    let classes = poem::resolve_classes(&trace).expect("every merge resolves");
    let text = classes
        .class_of(
            trace
                .facts
                .word(&Request::new(1, false).on_stream(Stream::Text))
                & classes.mask,
        )
        .expect("a text lane has a class");
    let image = classes
        .class_of(
            trace
                .facts
                .word(&Request::new(1, false).on_stream(Stream::Image))
                & classes.mask,
        )
        .expect("an image lane has a class");
    assert_ne!(text, image, "the two streams are two classes");

    let permutations: Vec<usize> = trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(
                node.op,
                Operation::Layout(Layout::PackRows { .. } | Layout::UnpackRows { .. })
            )
        })
        .map(|(at, _)| at)
        .collect();
    assert_eq!(permutations.len(), 4, "three packs and one unpack");
    for at in permutations {
        let mask = &classes.node_mask[at];
        assert!(
            mask.contains(text) && mask.contains(image),
            "node {at} ({:?}) runs in classes {mask:?}; a row permutation's window is the \
             SELECTION's packed rectangle, so it must run in every class its guard admits",
            trace.nodes[at].op
        );
    }
}
