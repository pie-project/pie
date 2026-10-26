//! Under `compiler.synth_anchors`, a matmul and the pointwise ops over its
//! product form an anchor (`fuse::anchor`): one kernels-ir anchored region,
//! its contraction the matmul and its epilogue the ops, verified pointwise
//! over the product tile and folded into one `fused.anchored` op where the
//! backend serves it. The product keeps its accumulator through a bias
//! alone, and the interpreter lands what the ops do over it.

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

/// `a · wᵀ` summed in f64 and rounded to bf16, as the program lands a
/// linear's output and an epilogue that only adds a bias does not read it.
fn landed(a: &[f32], w: &[f32], mnk: [usize; 3]) -> Vec<f32> {
    product(a, w, mnk).into_iter().map(round_bf16).collect()
}

/// `a · wᵀ` summed in f64 and rounded to f32 once, as the interpreter sums
/// a contraction: an epilogue that only adds a bias reads it so.
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
    assert_eq!(c.acc, Some(Dtype::Bf16), "its halves read as their projections land them");

    let [m, n, k] = [2usize, N as usize, K as usize];
    let (a, w) = (values(m * k, 4), values(2 * n * k, 5));
    let got = eval(&anchor.anchored, m, &a, &w, &[], &[]).expect("runs");
    let (g, u) = (
        landed(&a, &w[..n * k], [m, n, k]),
        landed(&a, &w[n * k..], [m, n, k]),
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
            Window::of(qw + kvw, kvw),
            Window::of(0, qw),
            Window::of(qw, kvw),
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
    let y = landed(&a, &w, [m, n as usize, k]);
    let v: Vec<f32> = (0..m)
        .flat_map(|r| y[r * n as usize + (qw + kvw) as usize..(r + 1) * n as usize].to_vec())
        .collect();
    assert_eq!(got[0], v, "v lands its products as the program does");
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

/// A KV append the qkv prologue ends in folds into its anchor: k and v
/// each a paged result (the cache's keys and values, at the rows the slot
/// mapping names, its inputs one i32 a row), the cache the op's, q its one
/// value landed.
#[test]
fn a_kv_append_lands_k_and_v_in_the_caches_pages() {
    use kernels_ir::{CacheDtype, CachePlane, Paged};
    use poem::ops::gather;
    use poem_ir::Turn;
    const D: u32 = 16;
    let (qw, kvw) = (2 * D, D);
    let build = |_: bool| {
        trace(|p| {
            let x = p.rows("x", K);
            let w = p.weight("qkv", [u64::from(qw + 2 * kvw), K]);
            let packed = linear::matmul(&x, &w);
            let (q, k, v) = gather::split_qkv(&packed, qw, kvw);
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
            let cache = p.cache("cache");
            let (page, offset) = (p.indices("write_page"), p.indices("write_offset"));
            attn::kv_append(&k.expect("k turned"), Some(&v), cache, &page, &offset);
            p.export("q", &q);
        })
    };
    let mut t = build(false);
    // the cache's planes as a model's trace states them: k's and v's rows
    for row in &mut t.caches {
        if let poem_ir::CacheRow::Kv { planes, .. } = row {
            *planes = vec![u64::from(kvw), u64::from(kvw)];
        }
    }
    let formed = anchors(&t, &all);
    let anchor = the_one(&formed);
    assert_eq!(
        anchor.nodes,
        [0, 1, 2, 3],
        "the product, split, turn and append"
    );
    assert!(anchor.cache.is_some() && !anchor.landed);
    assert_eq!(anchor.outs.len(), 1, "q alone a value");
    let epi = anchor.anchored.epilogue.as_ref().expect("an epilogue");
    assert_eq!(epi.inputs, [Input::Row, Input::Row, Input::Row]);
    assert_eq!(
        anchor.anchored.paged,
        [
            None,
            Some(Paged {
                plane: CachePlane::Keys,
                dtype: CacheDtype::Bf16,
                page: 1,
                slot: 2
            }),
            Some(Paged {
                plane: CachePlane::Values,
                dtype: CacheDtype::Bf16,
                page: 1,
                slot: 2
            }),
        ]
    );
    // The fold: one op, the cache its input, nothing past it but q.
    let (folded, _) = fold(t, &all);
    assert_eq!(folded.nodes.len(), 1, "{:#?}", folded.nodes);
    assert!(matches!(
        folded.nodes[0].op,
        Operation::Fused(Fused::Anchored { cache: Some(_), .. })
    ));
}

/// Qwen 3.5's gated attention prologue: the packed q|gate piece cut a head
/// at a time (q's and the gate's heads interleaved), q and k normed and
/// turned, k and v paged. q and the gate land head-major windows of the
/// product (a head of every two), k and v the cache's planes.
#[test]
fn a_gated_qs_heads_are_head_major_windows_of_the_product() {
    use kernels_ir::Window;
    use poem::ops::gather;
    use poem_ir::Turn;
    const D: u32 = 16;
    let (qg, kvw) = (4 * D, D);
    let mut t = trace(|p| {
        let packed = linear::matmul(
            &p.rows("x", K),
            &p.weight("qkv", [u64::from(qg + 2 * kvw), K]),
        );
        let (q, k, v) = gather::split_qkv(&packed, qg, kvw);
        let (q, gate) = gather::split_q_gate(&q, D);
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
        let cache = p.cache("cache");
        let (page, offset) = (p.indices("write_page"), p.indices("write_offset"));
        attn::kv_append(&k.expect("k turned"), Some(&v), cache, &page, &offset);
        p.export("q", &q);
        p.export("gate", &gate);
    });
    for row in &mut t.caches {
        if let poem_ir::CacheRow::Kv { planes, .. } = row {
            *planes = vec![u64::from(kvw), u64::from(kvw)];
        }
    }
    let formed = anchors(&t, &all);
    let anchor = the_one(&formed);
    assert_eq!(
        anchor.nodes,
        [0, 1, 2, 3, 4],
        "product, splits, turn and append"
    );
    let windows = &anchor.anchored.windows;
    assert_eq!(windows[..2].len(), 2);
    assert!(
        windows.contains(&Window::strided(D, 2 * D, D, 2 * D)),
        "the gate: {windows:?}"
    );
    assert!(
        windows.contains(&Window::strided(0, 2 * D, D, 2 * D)),
        "q: {windows:?}"
    );
    assert_eq!(anchor.anchored.verify(), Ok(()));
}

/// The gated q's head-major windows land what Qwen 3.5's ops do over them:
/// q's heads normed (plus-one weights) and turned on a partial rotary, the
/// gate's heads its products as they are summed.
#[test]
fn a_gated_qs_windows_land_its_norms_and_partial_turn() {
    use kernels_ir::Window;
    use poem::ops::gather;
    use poem_ir::Turn;
    const D: u32 = 16;
    const ROTARY: u32 = 8;
    const SPAN: u32 = 4;
    const THETA: f32 = 1e4;
    let (qg, kvw) = (4 * D, D);
    let t = trace(|p| {
        let packed = linear::matmul(
            &p.rows("x", K),
            &p.weight("qkv", [u64::from(qg + 2 * kvw), K]),
        );
        let (q, k, _) = gather::split_qkv(&packed, qg, kvw);
        let (q, gate) = gather::split_q_gate(&q, D);
        let q =
            elemwise::rmsnorm_per_head_plus_one(&q, &p.weight("q_norm", [u64::from(D)]), D, 1e-6);
        let k =
            elemwise::rmsnorm_per_head_plus_one(&k, &p.weight("k_norm", [u64::from(D)]), D, 1e-6);
        let (q, k) = elemwise::rope(
            &q,
            &p.indices("positions"),
            D,
            THETA,
            Turn::Partial {
                rotary_dim: ROTARY,
                span: SPAN,
            },
            Some(&k),
            None,
        );
        p.export("q", &q);
        p.export("k", &k.expect("k turned"));
        p.export("gate", &gate);
    });
    let formed = anchors(&t, &all);
    let anchor = the_one(&formed);
    let windows = &anchor.anchored.windows;
    let at = |w: Window| {
        windows
            .iter()
            .position(|&x| x == w)
            .unwrap_or_else(|| panic!("{w:?} in {windows:?}"))
    };
    let (q_at, gate_at) = (
        at(Window::strided(0, 2 * D, D, 2 * D)),
        at(Window::strided(D, 2 * D, D, 2 * D)),
    );
    let epi = anchor.anchored.epilogue.as_ref().expect("an epilogue");
    assert_eq!(epi.inputs, [Input::Head(D), Input::Head(D), Input::Row]);
    let n = (qg + 2 * kvw) as usize;
    let [m, k] = [3usize, K as usize];
    let (a, w) = (values(m * k, 6), values(n * k, 7));
    let (q_norm, positions) = (values(D as usize, 8), [0.0f32, 3.0, 17.0]);
    let given = [
        Given::Column(q_norm.clone()),
        Given::Column(values(D as usize, 9)),
        Given::Row(positions.to_vec()),
    ];
    let got = eval(&anchor.anchored, m, &a, &w, &[], &given).expect("runs");
    let y = landed(&a, &w, [m, n, k]);
    let d = D as usize;
    for row in 0..m {
        for head in 0..2 {
            let from = row * n + head * 2 * d;
            // the gate's head: its products as the program lands them
            let gate = &got[gate_at][row * 2 * d + head * d..][..d];
            assert_eq!(
                gate,
                &y[from + d..from + 2 * d],
                "row {row} head {head}'s gate"
            );
            // q's head: normed in f64, then turned
            let x = &y[from..from + d];
            let rms = (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / f64::from(D) + 1e-6)
                .sqrt()
                .recip();
            let normed: Vec<f64> = (0..d)
                .map(|i| f64::from(x[i]) * rms * (1.0 + f64::from(q_norm[i])))
                .collect();
            let mut want = normed.clone();
            for i in 0..(ROTARY / 2) as usize {
                let j = i + SPAN as usize;
                let angle = f64::from(positions[row])
                    * f64::from(THETA).powf(-(i as f64) / f64::from(SPAN));
                let (c, s) = (angle.cos(), angle.sin());
                want[i] = normed[i] * c - normed[j] * s;
                want[j] = normed[j] * c + normed[i] * s;
            }
            let q = &got[q_at][row * 2 * d + head * d..][..d];
            for i in 0..d {
                assert!(
                    (f64::from(q[i]) - want[i]).abs() <= want[i].abs() / 128.0 + 1e-2,
                    "row {row} head {head} column {i}: {} vs {}",
                    q[i],
                    want[i]
                );
            }
        }
    }
}
