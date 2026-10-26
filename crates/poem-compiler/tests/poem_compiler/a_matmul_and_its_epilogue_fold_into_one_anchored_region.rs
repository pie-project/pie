//! Under `compiler.synth_anchors`, a matmul and the pointwise ops over its
//! product form an anchor (`fuse::anchor`): one kernels-ir anchored region,
//! its contraction the matmul and its epilogue the ops, verified pointwise
//! over the product tile and folded into one `fused.anchored` op where the
//! backend serves it. The product keeps its accumulator through the
//! epilogue, and the interpreter lands what the ops do over it.

use kernels_ir::anchor::{Given, eval};
use kernels_ir::skeleton::round_bf16;
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

/// A packed qkv product's attention prologue as Qwen3 states it: the split
/// into q, k and v, q's and k's heads normed by their own weights and turned
/// at the rows' positions. One anchor of each head's span: the split's
/// pieces the results' windows, each head's weight a head's vector, the
/// position one a row; the cache write after it reads the pieces it lands.
#[test]
fn a_packed_qkvs_split_norms_and_turn_are_one_epilogue_of_windows() {
    use kernels_ir::Window;
    use poem::ops::gather;
    use poem_ir::Turn;
    const D: u32 = 16;
    let (qw, kvw) = (2 * D, D);
    let t = trace(|p| {
        let packed = linear::matmul(
            &p.rows("x", K),
            &p.weight("qkv", [u64::from(qw + 2 * kvw), K]),
        );
        let (q, k, v) = gather::split_qkv(&packed, qw, kvw);
        let q = elemwise::rmsnorm_per_head(&q, &p.weight("q_norm", [u64::from(D)]), D, 1e-6);
        let k = elemwise::rmsnorm_per_head(&k, &p.weight("k_norm", [u64::from(D)]), D, 1e-6);
        let positions = p.indices("positions");
        let (q, k) = elemwise::rope(
            &q,
            &positions,
            D,
            1e6,
            Turn::Full { interleaved: false },
            Some(&k),
            None,
        );
        let k = k.expect("k turned");
        let cache = p.cache("cache");
        let (page, offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
        attn::kv_append(&k, Some(&v), cache, &page, &offset);
        p.export("q", &q);
    });
    let formed = anchors(&t, &all);
    let anchor = the_one(&formed);
    assert_eq!(anchor.nodes, [0, 1, 2, 3, 4], "the cache write stays apart");
    let epi = anchor.anchored.epilogue.as_ref().expect("an epilogue");
    assert_eq!(
        epi.inputs,
        [Input::Head(D), Input::Head(D), Input::Row],
        "each head's weight, then the rows' positions"
    );
    let n = qw + 2 * kvw;
    assert_eq!(epi.span(n), D, "a head's norm and turn: one head's span");
    // v, then the turned q and k: each its own piece of the tile.
    assert_eq!(
        anchor.anchored.windows,
        [
            Window {
                from: qw + kvw,
                width: kvw
            },
            Window { from: 0, width: qw },
            Window {
                from: qw,
                width: kvw
            },
        ]
    );
    let [m, k] = [3usize, K as usize];
    let (a, w) = (values(m * k, 6), values(n as usize * k, 7));
    let given = [
        Given::Column(values(D as usize, 8)),
        Given::Column(values(D as usize, 9)),
        Given::Row(vec![0.0, 3.0, 17.0]),
    ];
    let got = eval(&anchor.anchored, m, &a, &w, &[], &given).expect("runs");
    let y = product(&a, &w, [m, n as usize, k]);
    let v: Vec<f32> = (0..m)
        .flat_map(|r| y[r * n as usize + (qw + kvw) as usize..(r + 1) * n as usize].to_vec())
        .collect();
    assert_eq!(got[0], v, "v lands its products as they are summed");
    // At position 0 nothing turns: q's heads are their norms.
    let q0 = &got[1][..qw as usize];
    let head = &y[..D as usize];
    let rms = (head.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / f64::from(D) + 1e-6)
        .sqrt()
        .recip();
    let w0 = values(D as usize, 8);
    for i in 0..D as usize {
        let want = round_bf16(round_bf16((f64::from(head[i]) * rms) as f32) * w0[i]);
        assert!(
            (q0[i] - want).abs() <= want.abs() / 64.0 + 1e-3,
            "{i}: {} vs {want}",
            q0[i]
        );
    }
}

/// A norm across a piece's whole row (not its heads) reads past one head's
/// span: the anchor stops at the split before it.
#[test]
fn a_rows_norm_over_a_piece_is_no_ones_span() {
    use poem::ops::gather;
    const D: u32 = 16;
    let t = trace(|p| {
        let packed = linear::matmul(&p.rows("x", K), &p.weight("qkv", [u64::from(4 * D), K]));
        let (q, k, v) = gather::split_qkv(&packed, 2 * D, D);
        let q = elemwise::rmsnorm(&q, &p.weight("q_norm", [u64::from(2 * D)]), 1e-6);
        p.export("q", &q);
        p.export("k", &k);
        p.export("v", &v);
    });
    let formed = anchors(&t, &all);
    let anchor = the_one(&formed);
    assert_eq!(anchor.nodes, [0, 1], "the split alone");
    assert_eq!(anchor.anchored.windows.len(), 3);
}

/// An RMS norm whose row a product's epilogue reads (a SwiGLU's packed
/// bank) folds into the anchor: its operand the norm's input, which the
/// residual stream still reads, each row's scale the epilogue's
/// `Input::Norm`, the gain the load's to fold into the bank. Over the bank
/// as `fold_bank` declares it, the interpreter lands the unfolded program
/// within each side's rounding. A bare product, a normed row something
/// else reads, and the fold off keep the norm its own op.
#[test]
fn an_rms_norm_before_an_epilogued_product_folds_into_its_anchor() {
    use kernels_ir::anchor::{fold_bank, norm_scales};
    use kernels_ir::{Gain, RowNorm};
    use poem_compiler::fuse::anchor::anchors_with;
    let build = |bare: bool, shared: bool| {
        trace(|p| {
            let x = p.rows("x", K);
            let normed = elemwise::rmsnorm(&x, &p.weight("norm", [K]), 1e-6);
            let packed = linear::matmul(&normed, &p.weight("gate_up", [2 * N, K]));
            if bare {
                p.export("packed", &packed);
            } else {
                let y = linear::glu(&packed, GluAct::Swiglu, Some(N as u32), None, None, None);
                p.export("y", &y);
            }
            if shared {
                let _ = elemwise::residual_add(&normed, &p.rows("other", K));
            }
            let _ = elemwise::residual_add(&x, &p.rows("stream", K));
        })
    };
    let t = build(false, false);
    let formed = anchors_with(&t, &all, Some(&|_| true));
    let anchor = &formed[0];
    assert_eq!(anchor.nodes, [0, 1, 2], "the norm, the product and its GLU");
    let Operation::Elemwise(poem_ir::Elemwise::Rmsnorm { x, .. }) = t.nodes[0].op else {
        panic!("{:?}", t.nodes[0].op);
    };
    assert_eq!(anchor.act, x, "the norm's input");
    assert!(anchor.gain.is_some(), "the gain the load folds");
    let norm = anchor.anchored.norm.expect("a norm");
    assert_eq!(norm.gain, Gain::Scale);
    let epi = anchor.anchored.epilogue.as_ref().expect("an epilogue");
    assert_eq!(epi.inputs.last(), Some(&Input::Norm));

    let [m, n, k] = [3usize, N as usize, K as usize];
    let (a, w, g) = (values(m * k, 1), values(2 * n * k, 2), values(k, 3));
    let bank = fold_bank(&norm, &w, &g);
    let got = eval(&anchor.anchored, m, &a, &bank, &[], &[]).expect("runs");
    // The unfolded program: the normed row rounded to bf16, then the GLU
    // over its products.
    let r = norm_scales(
        &RowNorm {
            eps: 1e-6,
            gain: Gain::None,
        },
        &a,
        k,
    );
    let normed: Vec<f32> = (0..m * k)
        .map(|i| round_bf16(a[i] * r[i / k] * g[i % k]))
        .collect();
    let (gate, up) = (
        product(&normed, &w[..n * k], [m, n, k]),
        product(&normed, &w[n * k..], [m, n, k]),
    );
    // Each side rounds each term once (the normed row, or the folded bank):
    // a product off by a bf16 place of its terms' magnitudes' sum, carried
    // through the GLU (silu's slope at most 1.1), then the GLU's roundings.
    let place = 2f32.powi(-8);
    let magnitude = |bank: &[f32], i: usize| -> f32 {
        let (row, col) = (i / n, i % n);
        (0..k)
            .map(|j| (normed[row * k + j] * bank[col * k + j]).abs())
            .sum()
    };
    for i in 0..m * n {
        let silu = gate[i] / (1.0 + (-gate[i]).exp());
        let want = silu * up[i];
        let (dg, du) = (
            place * (magnitude(&w[..n * k], i) + gate[i].abs()),
            place * (magnitude(&w[n * k..], i) + up[i].abs()),
        );
        let bound = 1.1 * dg * up[i].abs() + silu.abs() * du + 2.0 * place * want.abs() + 1e-3;
        assert!(
            (got[0][i] - want).abs() <= bound,
            "{i}: {} vs {want} (within {bound})",
            got[0][i]
        );
    }

    for (bare, shared, norms, why) in [
        (true, false, true, "a bare product"),
        (false, true, true, "a normed row read past the product"),
        (false, false, false, "the fold off"),
    ] {
        let t = build(bare, shared);
        let formed = anchors_with(&t, &all, norms.then_some(&|_| true));
        assert!(
            formed
                .iter()
                .all(|a| a.anchored.norm.is_none() && !a.nodes.contains(&0)),
            "{why}: {formed:#?}"
        );
    }
}
