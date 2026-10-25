//! Under `compiler.synth_anchors`, a matmul and the pointwise ops over its
//! product form an anchor (`fuse::anchor`): one kernels-ir anchored region,
//! its contraction the matmul and its epilogue the ops, verified pointwise
//! over the product tile and folded into one `fused.anchored` op where the
//! backend serves it. The product keeps its accumulator through the
//! epilogue, and the interpreter lands what the ops do over it.

use kernels_ir::anchor::{Given, eval};
use kernels_ir::frame::round_bf16;
use kernels_ir::{Anchored, Input, Layout};
use poem::ops::{attn, elemwise, linear};
use poem::pattern::{Pattern, template};
use poem_compiler::fuse::anchor::{Anchor, anchors, fold};
use poem_ir::{Dtype, Fused, GluAct, Operation, Trace};

const K: u64 = 32;
const N: u64 = 16;

fn trace(build: impl FnOnce(&Pattern)) -> Trace {
    template(build).trace
}

/// Every anchor kernels-ir verifies.
fn all(_: &Anchored, _: &[Dtype], _: &[Dtype]) -> bool {
    true
}

fn the_one(formed: &[Anchor]) -> &Anchor {
    assert_eq!(formed.len(), 1, "{formed:#?}");
    &formed[0]
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| round_bf16(((i * 37 + seed * 11) % 23) as f32 / 8.0 - 1.4))
        .collect()
}

/// `a · wᵀ` summed in f64 and rounded to f32 once, as the interpreter sums
/// a contraction: the epilogue reads it so, no op between them rounding it.
fn product(a: &[f32], w: &[f32], [m, n, k]: [usize; 3]) -> Vec<f32> {
    (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            let dot = (0..k).fold(0f64, |s, j| {
                s + f64::from(a[row * k + j]) * f64::from(w[col * k + j])
            });
            dot as f32
        })
        .collect()
}

#[test]
fn a_bias_and_a_soft_cap_are_the_products_epilogue() {
    let cap = 30.0;
    let t = trace(|p| {
        let y = linear::matmul(&p.rows("x", K), &p.weight("w", [N, K]));
        let b = elemwise::add_bias(&p.weight("bias", [N]), &y);
        let _ = attn::logit_softcap(&b, cap);
    });
    let formed = anchors(&t, &all);
    let anchor = the_one(&formed);
    assert_eq!(anchor.nodes, [0, 1, 2]);
    let c = &anchor.anchored.contraction;
    assert_eq!((c.n, c.k, c.layout), (N as u32, K as u32, Layout::Dense));
    assert_eq!(
        c.acc, None,
        "the product keeps its accumulator through the bias"
    );
    let epi = anchor.anchored.epilogue.as_ref().expect("an epilogue");
    assert_eq!(epi.inputs, [Input::Column], "the bias a vector a column");
    assert_eq!(anchor.outs.len(), 1, "the capped logits alone leave");

    let [m, n, k] = [3usize, N as usize, K as usize];
    let (a, w, bias) = (values(m * k, 1), values(n * k, 2), values(n, 3));
    let got = eval(
        &anchor.anchored,
        m,
        &a,
        &w,
        &[],
        &[Given::Column(bias.clone())],
    )
    .expect("the anchored region runs");
    let y = product(&a, &w, [m, n, k]);
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let biased = round_bf16(y[i] + bias[i % n]);
            round_bf16(cap * (biased / cap).tanh())
        })
        .collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got[0]), bits(&want));
}

#[test]
fn a_residual_add_joins_and_the_norm_after_it_does_not() {
    let t = trace(|p| {
        let y = linear::matmul(&p.rows("x", K), &p.weight("o", [N, K]));
        let h = elemwise::residual_add(&y, &p.rows("stream", N));
        let normed = elemwise::rmsnorm(&h, &p.weight("norm", [N]), 1e-6);
        let _ = linear::matmul(&normed, &p.weight("next", [N, N]));
        let _ = linear::matmul(&h, &p.weight("later", [N, N]));
    });
    let formed = anchors(&t, &all);
    let first = formed.first().expect("an anchor");
    assert_eq!(first.nodes, [0, 1], "a norm sums across the row");
    let epi = first.anchored.epilogue.as_ref().expect("an epilogue");
    assert_eq!(epi.inputs, [Input::Tile], "the stream a tile of the rows");
    assert_eq!(first.outs.len(), 1, "the stream leaves");
}

#[test]
fn a_packed_glu_reads_the_banks_two_parts() {
    let t = trace(|p| {
        let packed = linear::matmul(&p.rows("x", K), &p.weight("gate_up", [2 * N, K]));
        let _ = linear::glu(&packed, GluAct::GegluTanh, Some(N as u32), None, None, None);
    });
    let formed = anchors(&t, &all);
    let anchor = the_one(&formed);
    let c = &anchor.anchored.contraction;
    assert_eq!((c.n, c.layout), (N as u32, Layout::Packed(2)));

    let [m, n, k] = [2usize, N as usize, K as usize];
    let (a, w) = (values(m * k, 4), values(2 * n * k, 5));
    let got = eval(&anchor.anchored, m, &a, &w, &[], &[]).expect("runs");
    let (g, u) = (
        product(&a, &w[..n * k], [m, n, k]),
        product(&a, &w[n * k..], [m, n, k]),
    );
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let x = g[i];
            let inner = 0.797_884_6 * (x + 0.044_715 * (x * x * x));
            round_bf16(0.5 * x * (1.0 + inner.tanh()) * u[i])
        })
        .collect();
    for (got, want) in got[0].iter().zip(&want) {
        assert!((got - want).abs() <= want.abs() / 128.0, "{got} vs {want}");
    }
}

#[test]
fn a_product_read_past_its_epilogue_forms_no_anchor() {
    let t = trace(|p| {
        let y = linear::matmul(&p.rows("x", K), &p.weight("w", [N, K]));
        let _ = elemwise::add_bias(&p.weight("bias", [N]), &y);
        let _ = linear::matmul(&y, &p.weight("again", [N, N]));
    });
    assert!(anchors(&t, &all).is_empty());
}

#[test]
fn a_fold_lands_one_op_where_the_backend_serves_it() {
    let build = |p: &Pattern| {
        let y = linear::matmul(&p.rows("x", K), &p.weight("w", [N, K]));
        let _ = elemwise::add_bias(&p.weight("bias", [N]), &y);
    };
    let t = trace(build);
    let (folded, held) = fold(t.clone(), &all);
    assert_eq!(folded.nodes.len(), 1);
    let Operation::Fused(Fused::Anchored { ins, key, outs, .. }) = &folded.nodes[0].op else {
        panic!("{:?}", folded.nodes[0].op);
    };
    assert_eq!((ins.len(), outs.len()), (1, 1));
    assert!(held.contains_key(key));
    // One the backend does not serve keeps its ops.
    let (kept, none) = fold(t.clone(), &|_, _, _| false);
    assert!(none.is_empty());
    assert_eq!(kept.nodes.len(), t.nodes.len());
}
