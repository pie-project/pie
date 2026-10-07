mod common;

use poem_compiler::Placement;
use poem_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Request, Value, Weight, ops, seam,
    trace_hybrid,
};
use poem_ir::{Attention, Elementwise, Operation, RaggedMask};

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
const LAYERS: u32 = 2;
const MAX_LEN: u32 = 512;
const NUM_BUCKETS: u32 = 32;

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
            let table = ops::elemwise::relative_bucket_bias(
                inputs.recorder(),
                &Weight::sym(
                    format!("l{l}.rel_bias"),
                    [u64::from(NUM_BUCKETS), u64::from(HEADS)],
                    Dtype::Bf16,
                ),
                MAX_LEN,
                NUM_BUCKETS,
                128.0,
                true,
            );
            let norm = Weight::sym(format!("l{l}.norm"), [u64::from(WIDTH)], Dtype::Bf16);
            let h = ops::elemwise::rmsnorm(&x, &norm, 1e-6);
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
        }
        seam::at(seam::HIDDEN, &[&x]);
        x
    }
}

#[test]
fn the_tables_are_fixed_rectangles_alive_through_their_attention() {
    for platform in common::PLATFORMS {
        let trace = trace_hybrid("encoder", &Encoder, platform);
        assert!(trace.caches.is_empty());

        let compiled = common::bake(&trace).unwrap_or_else(|e| panic!("{platform:?}: {e}"));
        assert!(
            compiled.classes.dead.is_empty(),
            "{platform:?}: every node is demanded: {:?}",
            compiled.classes.dead
        );

        let mut seen = 0;
        for (at, node) in trace.nodes.iter().enumerate() {
            let Operation::Elementwise(Elementwise::RelativeBucketBias { y, .. }) = &node.op else {
                continue;
            };
            seen += 1;
            let slot = y.0 as usize;
            let Placement::Arena { bytes, .. } = compiled.arena.placements[slot] else {
                panic!(
                    "{platform:?}: the table is an arena rectangle, not {:?}",
                    compiled.arena.placements[slot]
                );
            };
            assert_eq!(
                bytes,
                u64::from(HEADS) * (2 * u64::from(MAX_LEN) - 1) * 4,
                "{platform:?}: the table is heads · (2·max_len − 1) f32, no ceiling applied"
            );
            let reader = trace
                .nodes
                .iter()
                .position(|node| {
                    matches!(
                        &node.op,
                        Operation::Attention(Attention::Ragged {
                            mask: RaggedMask::RelativeBias { table, .. },
                            ..
                        }) if table == y
                    )
                })
                .expect("the attention that reads this table");
            assert!(reader > at, "the table is computed before it is read");
            let span = compiled.arena.spans[slot].expect("a rectangle has a life");
            assert!(
                (span.first as usize) <= at && (span.last as usize) >= reader,
                "{platform:?}: the table's life {span:?} covers node {at} through {reader}"
            );
        }
        assert_eq!(seen, LAYERS as usize, "{platform:?}: one table per layer");
    }
}
