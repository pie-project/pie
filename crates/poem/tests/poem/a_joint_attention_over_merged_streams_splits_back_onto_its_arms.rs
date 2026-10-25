use poem::fact;
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, RaggedMask, Request, Value, Weight, ops,
    seam, trace_hybrid,
};
use poem_ir::{Attn, Def, Elemwise, GeomKind, Guard, Operation, RuntimeInput, Selection};

const WIDTH: u32 = 32;
const HEAD_DIM: u32 = 8;

struct DoubleBlock;

impl ForwardHybrid for DoubleBlock {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        inputs
            .recorder()
            .declare_switch("stream", &["text", "image", "video", "audio"]);
        let (txt, img) = (
            inputs.on(fact::switch_is("stream", "text")),
            inputs.on(!fact::switch_is("stream", "text")),
        );
        let w = |name: &str| Weight::sym(name, [u64::from(WIDTH), u64::from(WIDTH)], Dtype::Bf16);
        let x_txt = txt.input("text", Dtype::Bf16, poem_ir::InputAxis::Row, WIDTH, None);
        let x_img = img.input("image", Dtype::Bf16, poem_ir::InputAxis::Row, WIDTH, None);
        let project = |x: &Value, prefix: &str| {
            (
                ops::linear::matmul(x, &w(&format!("{prefix}.q"))),
                ops::linear::matmul(x, &w(&format!("{prefix}.k"))),
                ops::linear::matmul(x, &w(&format!("{prefix}.v"))),
            )
        };
        let (qt, kt, vt) = project(&x_txt, "txt");
        let (qi, ki, vi) = project(&x_img, "img");
        let q = Value::merge(vec![qt, qi]);
        let k = Value::merge(vec![kt, ki]);
        let v = Value::merge(vec![vt, vi]);
        let perm = inputs.row_permutation();
        let indptr = inputs.group_indptr();
        let o = ops::attn::ragged(
            &ops::gather::pack_rows(&q, &perm),
            &ops::gather::pack_rows(&k, &perm),
            &ops::gather::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            0.35,
            RaggedMask::GroupBlockDiagonal,
        );
        let o = ops::gather::unpack_rows(&o, &perm);
        let (o_txt, o_img) = (
            o.on(fact::switch_is("stream", "text")),
            o.on(!fact::switch_is("stream", "text")),
        );
        let y_txt = ops::linear::matmul(&o_txt, &w("txt.o"));
        let y_img = ops::linear::matmul(&o_img, &w("img.o"));
        let r_txt = ops::elemwise::residual_add(&y_txt, &x_txt);
        let r_img = ops::elemwise::residual_add(&y_img, &x_img);
        let out = Value::merge(vec![r_txt, r_img]);
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

#[test]
fn the_joint_attention_traces_and_its_answer_splits_back_onto_the_arms() {
    let trace = trace_hybrid("double", &DoubleBlock, Platform::Cuda);
    let text = Guard::Fact(0);
    let image = !text.clone();

    let ragged = trace
        .nodes
        .iter()
        .find(|node| matches!(node.op, Operation::Attn(Attn::Ragged { .. })))
        .expect("one joint attention");
    assert!(
        ragged
            .guard
            .equivalent(&Guard::or(text.clone(), image.clone()))
    );
    let Operation::Attn(Attn::Ragged {
        q_indptr,
        kv_indptr,
        mask,
        ..
    }) = &ragged.op
    else {
        unreachable!()
    };
    assert_eq!(*mask, RaggedMask::GroupBlockDiagonal);
    assert_eq!(q_indptr, kv_indptr, "self-attention: one CSR both sides");

    assert_eq!(
        trace.values[q_indptr.0 as usize].def,
        Def::Input(RuntimeInput::Geometry {
            space: 0,
            kind: GeomKind::GroupIndptr {
                select: Selection::ALL
            }
        })
    );

    let folds: Vec<&Guard> = trace
        .nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Elemwise(Elemwise::ResidualAdd { .. })))
        .map(|node| &node.guard)
        .collect();
    assert_eq!(
        folds,
        vec![&text, &image],
        "one fold per arm, spelled as the arm"
    );
    let projections: Vec<&Guard> = trace
        .nodes
        .iter()
        .rev()
        .filter(|node| {
            matches!(
                node.op,
                Operation::Linear(poem_ir::Linear::Matmul {
                    quant: None,
                    logits: false,
                    ..
                })
            )
        })
        .take(2)
        .map(|node| &node.guard)
        .collect();
    assert_eq!(projections, vec![&image, &text]);

    let classes = poem::resolve_classes(&trace).expect("every merge resolves");
    let word = trace
        .facts
        .word(&Request::new(1, false).switched("stream", "text"))
        & classes.mask;
    let class = classes.class_of(word).expect("a text lane has a class");
    for row in 0..classes.merges.len() {
        assert_eq!(
            classes.merge_arm[row][class],
            Some(0),
            "arm 0 is the text arm"
        );
    }
}
