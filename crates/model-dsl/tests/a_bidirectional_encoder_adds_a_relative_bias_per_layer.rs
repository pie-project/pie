use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops, seam,
    trace_hybrid,
};
use model_ir::{Attention, Def, Dim, Elementwise, Guard, Operands, Operation, RaggedMask, Ty};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

const WIDTH: u32 = 64;
const HEADS: u32 = 4;
const HEAD_DIM: u32 = 16;
const INNER: u32 = 96;
const LAYERS: u32 = 2;
const MAX_LEN: u32 = 512;
const NUM_BUCKETS: u32 = 32;
const MAX_DISTANCE: f32 = 128.0;

struct Encoder;

impl ForwardHybrid for Encoder {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };
        let mut x = inputs.latents(0, WIDTH, Dtype::Bf16);
        let perm = inputs.row_permutation();
        let indptr = inputs.lane_indptr();
        let layers: Vec<u32> = (0..LAYERS).collect();
        for (l, _) in inputs.walk_layers(&layers) {
            let ln =
                |name: &str| Weight::sym(format!("l{l}.{name}"), [u64::from(WIDTH)], Dtype::Bf16);
            let table = ops::elemwise::relative_bucket_bias(
                inputs.recorder(),
                &Weight::sym(
                    format!("l{l}.rel_bias"),
                    [u64::from(NUM_BUCKETS), u64::from(HEADS)],
                    Dtype::Bf16,
                ),
                MAX_LEN,
                NUM_BUCKETS,
                MAX_DISTANCE,
                true,
            );
            let h = ops::elemwise::rmsnorm(&x, &ln("attn_norm"), 1e-6);
            let q = ops::linear::matmul(&h, &w(&format!("l{l}.q"), HEADS * HEAD_DIM, WIDTH));
            let k = ops::linear::matmul(&h, &w(&format!("l{l}.k"), HEADS * HEAD_DIM, WIDTH));
            let v = ops::linear::matmul(&h, &w(&format!("l{l}.v"), HEADS * HEAD_DIM, WIDTH));
            let o = ops::attn::ragged(
                &ops::layout::pack_rows(&q, &perm),
                &ops::layout::pack_rows(&k, &perm),
                &ops::layout::pack_rows(&v, &perm),
                &indptr,
                &indptr,
                HEAD_DIM,
                1.0,
                ops::attn::relative_bias(&table, MAX_LEN),
            );
            let o = ops::layout::unpack_rows(&o, &perm);
            let y = ops::linear::matmul(&o, &w(&format!("l{l}.o"), WIDTH, HEADS * HEAD_DIM));
            x = ops::elemwise::residual_add(&x, &y);
            let h = ops::elemwise::rmsnorm(&x, &ln("mlp_norm"), 1e-6);
            let gate = ops::linear::matmul(&h, &w(&format!("l{l}.wi_0"), INNER, WIDTH));
            let up = ops::linear::matmul(&h, &w(&format!("l{l}.wi_1"), INNER, WIDTH));
            let act = ops::linear::mlp_geglu_tanh(&gate, &up);
            let y = ops::linear::matmul(&act, &w(&format!("l{l}.wo"), WIDTH, INNER));
            x = ops::elemwise::residual_add(&x, &y);
        }
        let out = ops::elemwise::rmsnorm(
            &x,
            &Weight::sym("final_norm", [u64::from(WIDTH)], Dtype::Bf16),
            1e-6,
        );
        seam::at(seam::HIDDEN, &[&out]);
        out
    }
}

#[test]
fn a_bidirectional_encoder_adds_a_relative_bias_per_layer_every_case() {
    two_layers_trace_with_one_table_each_and_the_attention_names_it();
    a_table_that_disagrees_with_its_max_len_or_is_not_f32_is_refused();
}

fn two_layers_trace_with_one_table_each_and_the_attention_names_it() {
    let trace = trace_hybrid("encoder", &Encoder, Platform::Cuda);
    assert!(trace.caches.is_empty(), "an encoder declares no kv space");
    assert!(
        !trace.seams.iter().any(|s| s.seam == seam::OUT.name),
        "no logits: the hidden seam is the readout"
    );
    assert!(trace.seams.iter().any(|s| s.seam == seam::HIDDEN.name));

    let tables: Vec<(usize, &model_ir::Node)> = trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(
                node.op,
                Operation::Elementwise(Elementwise::RelativeBucketBias { .. })
            )
        })
        .collect();
    assert_eq!(tables.len(), LAYERS as usize, "one table per layer");
    let mut table_ids = Vec::new();
    for (at, node) in &tables {
        let Operation::Elementwise(Elementwise::RelativeBucketBias {
            embedding,
            max_len,
            num_buckets,
            max_distance,
            bidirectional,
            y,
        }) = &node.op
        else {
            unreachable!()
        };
        assert_eq!(
            (*max_len, *num_buckets, *max_distance, *bidirectional),
            (MAX_LEN, NUM_BUCKETS, MAX_DISTANCE, true)
        );
        assert_eq!(
            node.guard,
            Guard::Always,
            "a plan constant is guarded Always"
        );
        assert!(
            matches!(trace.values[embedding.0 as usize].def, Def::Weight(_)),
            "the embedding is a weight"
        );
        assert_eq!(
            trace.values[embedding.0 as usize].ty,
            Ty::Tensor {
                shape: vec![
                    Dim::Const(u64::from(NUM_BUCKETS)),
                    Dim::Const(u64::from(HEADS))
                ],
                dtype: Dtype::Bf16
            }
        );
        assert_eq!(
            trace.values[y.0 as usize].ty,
            Ty::Tensor {
                shape: vec![
                    Dim::Const(u64::from(HEADS)),
                    Dim::Const(2 * u64::from(MAX_LEN) - 1)
                ],
                dtype: Dtype::F32
            },
            "the table is [heads, 2·max_len − 1] f32"
        );
        assert_eq!(trace.values[y.0 as usize].def, Def::Op(*at as u32));
        table_ids.push(*y);
    }

    let ragged: Vec<&model_ir::Node> = trace
        .nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Attention(Attention::Ragged { .. })))
        .collect();
    assert_eq!(ragged.len(), LAYERS as usize, "one attention per layer");
    for (node, table) in ragged.iter().zip(&table_ids) {
        let Operation::Attention(Attention::Ragged {
            mask,
            sm_scale,
            kv_heads,
            ..
        }) = &node.op
        else {
            unreachable!()
        };
        assert_eq!(
            *mask,
            RaggedMask::RelativeBias {
                table: *table,
                max_len: MAX_LEN
            },
            "the attention carries its own layer's table"
        );
        assert_eq!(*sm_scale, 1.0, "T5 does not scale its logits");
        assert_eq!(*kv_heads, HEADS);
        let mut inputs = Vec::new();
        node.op.inputs(&mut inputs);
        assert_eq!(inputs.len(), 6);
        assert_eq!(inputs[5], *table, "the table is the node's sixth operand");
    }
}

fn refusal(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    match std::panic::catch_unwind(f) {
        Ok(()) => panic!("the mask was accepted"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default(),
    }
}

fn a_table_that_disagrees_with_its_max_len_or_is_not_f32_is_refused() {
    struct Misshapen(u32, Dtype);
    impl ForwardHybrid for Misshapen {
        type Facts = NoFacts;
        fn caches(&self) -> HybridSpec {
            HybridSpec::new()
        }
        fn forward(&self, inputs: Input<NoFacts>) -> Value {
            let x = inputs.latents(0, HEADS * HEAD_DIM, Dtype::Bf16);
            let table = inputs.recorder().fresh(Ty::Tensor {
                shape: vec![Dim::Const(u64::from(HEADS)), Dim::Const(u64::from(self.0))],
                dtype: self.1,
            });
            let indptr = inputs.lane_indptr();
            let o = ops::attn::ragged(
                &x,
                &x,
                &x,
                &indptr,
                &indptr,
                HEAD_DIM,
                1.0,
                ops::attn::relative_bias(&table, MAX_LEN),
            );
            seam::at(seam::HIDDEN, &[&o]);
            o
        }
    }
    let narrow = refusal(|| {
        trace_hybrid(
            "narrow",
            &Misshapen(2 * MAX_LEN, Dtype::F32),
            Platform::Cuda,
        );
    });
    assert!(narrow.contains("2 · max_len − 1"), "{narrow}");
    let bf16 = refusal(|| {
        trace_hybrid(
            "bf16",
            &Misshapen(2 * MAX_LEN - 1, Dtype::Bf16),
            Platform::Cuda,
        );
    });
    assert!(bf16.contains("f32"), "{bf16}");
}
