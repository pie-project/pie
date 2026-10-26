//! The contraction skeletons (`linear::skeleton`) with an epilogue filled in land
//! what the anchored region states: each skeleton and each epilogue against
//! kernels-ir's interpreter (`anchor::eval`), within the tolerances below
//! (in bf16 ulps, as `elemwise::within` counts them), and the same bits on
//! a second fire.
//!
//! - **The contraction's order is the skeleton's.** The interpreter sums a
//!   product in f64 and rounds it to f32 once; a skeleton sums it in f32 in
//!   its own order (a warp's lanes, a block's slices, mma.sync's tiles), off
//!   by a few f32 ulps of the sum of its terms' magnitudes. That moves a
//!   product rounded to bf16 only where it lies that close to a rounding
//!   edge, by one ulp: [`PRODUCT`].
//! - **Each rounding point after carries it on**, one ulp more each, and an
//!   op that scales its input carries it scaled: a bias or a soft cap
//!   (`expf` within 2 f32 ulps), [`ONE_MORE`]; GEGLU, two products each an
//!   ulp off, its gate's through `gelu`'s slope (at most 1.13), [`GLU`]. The tuner's anchored
//! entry (`gemm::anchored`) lands the same on whichever candidate it picks.

use crate::common::linear_ref::t;
use crate::common::{Gpu, Lcg};
use crate::elemwise::within::{assert_within, deterministic};
use dtype::Dtype::{self, Bf16};
use kernels_cuda::linear::gemm;
use kernels_cuda::linear::skeleton::{
    self, Anchored, Contraction, Epilogue, In, Input, Layout, Pointwise, Quant, Skeleton,
};
use kernels_cuda::tuning::Tuning;
use kernels_ir::anchor::{Given, eval};
use kernels_ir::skeleton::round_bf16;
use kernels_ir::{Binary, Literal, Op, Region, Shape, Unary};

fn contraction(n: usize, k: usize, layout: Layout, acc: Option<Dtype>) -> Contraction {
    Contraction {
        n: n as u32,
        k: k as u32,
        layout,
        quant: Quant::Plain(Bf16),
        acc,
    }
}

/// What a skeleton is given and what the interpreter is: the epilogue's
/// region, its inputs, and the contraction's rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    Store,
    Bias,
    Softcap,
    Geglu,
    /// `(y · one + bias[n]) · s[m] + r[m][n]` and the bare product, over the
    /// product unrounded: every input kind, two results, the f32 tile.
    Every,
}

impl Case {
    const ALL: [Case; 5] = [
        Case::Store,
        Case::Bias,
        Case::Softcap,
        Case::Geglu,
        Case::Every,
    ];

    fn layout(self) -> Layout {
        if self == Case::Geglu {
            Layout::Packed(2)
        } else {
            Layout::Dense
        }
    }

    fn inputs(self) -> Vec<Input> {
        match self {
            Case::Bias => vec![Input::Column],
            Case::Every => vec![Input::One, Input::Column, Input::Row, Input::Tile],
            _ => Vec::new(),
        }
    }

    /// The anchored region at `n x k`, the region a skeleton's epilogue is
    /// printed from.
    fn anchored(self, n: usize, k: usize) -> Anchored {
        let acc = if self == Case::Every {
            None
        } else {
            Some(Bf16)
        };
        let plain = Anchored::plain(contraction(n, k, self.layout(), acc));
        let width = n as u32;
        let region = match self {
            Case::Store => return plain,
            Case::Bias => Region {
                width,
                params: vec![Shape::Row, Shape::Row],
                ops: vec![Op::Binary(Binary::Add, 0, 1)],
                results: vec![2],
                rounds: vec![(2, Bf16)],
            },
            Case::Softcap => Region {
                width,
                params: vec![Shape::Row],
                ops: vec![
                    Op::Const(Literal::F32(-2.0 / 30.0)),
                    Op::Binary(Binary::Mul, 0, 1),
                    Op::Unary(Unary::Exp, 2),
                    Op::Const(Literal::F32(1.0)),
                    Op::Binary(Binary::Add, 4, 3),
                    Op::Const(Literal::F32(2.0)),
                    Op::Binary(Binary::Div, 6, 5),
                    Op::Binary(Binary::Sub, 7, 4),
                    Op::Const(Literal::F32(30.0)),
                    Op::Binary(Binary::Mul, 9, 8),
                ],
                results: vec![10],
                rounds: vec![(10, Bf16)],
            },
            Case::Geglu => Region {
                width,
                params: vec![Shape::Row, Shape::Row],
                ops: vec![
                    Op::Binary(Binary::Mul, 0, 0),
                    Op::Binary(Binary::Mul, 2, 0),
                    Op::Const(Literal::F32(0.044_715)),
                    Op::Binary(Binary::Mul, 4, 3),
                    Op::Binary(Binary::Add, 0, 5),
                    Op::Const(Literal::F32(-2.0 * 0.797_884_6)),
                    Op::Binary(Binary::Mul, 7, 6),
                    Op::Unary(Unary::Exp, 8),
                    Op::Const(Literal::F32(1.0)),
                    Op::Binary(Binary::Add, 10, 9),
                    Op::Binary(Binary::Div, 0, 11),
                    Op::Binary(Binary::Mul, 12, 1),
                ],
                results: vec![13],
                rounds: vec![(13, Bf16)],
            },
            // y, one, bias, s, r
            Case::Every => Region {
                width,
                params: vec![
                    Shape::Row,
                    Shape::Scalar,
                    Shape::Row,
                    Shape::Scalar,
                    Shape::Row,
                ],
                ops: vec![
                    Op::Binary(Binary::Mul, 0, 1), // 5
                    Op::Binary(Binary::Add, 5, 2), // 6
                    Op::Binary(Binary::Mul, 6, 3), // 7
                    Op::Binary(Binary::Add, 7, 4), // 8
                ],
                results: vec![8, 0],
                rounds: vec![(6, Bf16), (8, Bf16)],
            },
        };
        Anchored {
            epilogue: Some(Pointwise {
                region,
                inputs: self.inputs(),
            }),
            ..plain
        }
    }
}

/// A product rounded to bf16, its sum in the skeleton's order.
const PRODUCT: f32 = 1.0;
/// A product's ulp and one rounding point after it.
const ONE_MORE: f32 = 2.0;
/// Two products' ulps through GEGLU, and its rounding.
const GLU: f32 = 3.0;

impl Case {
    /// How many bf16 ulps the case's results land from the interpreter.
    fn tolerance(self) -> f32 {
        match self {
            Case::Store => PRODUCT,
            Case::Bias | Case::Softcap | Case::Every => ONE_MORE,
            Case::Geglu => GLU,
        }
    }
}

/// The case's inputs at `m x n`: on the device, as the skeleton reads them, and
/// on the host, as the interpreter does.
fn inputs(gpu: &mut Gpu, lcg: &mut Lcg, case: Case, m: usize, n: usize) -> (Vec<In>, Vec<Given>) {
    let mut up = |count: usize| {
        let (raw, exact) = lcg.row(count);
        (gpu.up(&raw), exact)
    };
    let mut ins = Vec::new();
    let mut given = Vec::new();
    for input in case.inputs() {
        match input {
            Input::One => {
                let (at, v) = up(1);
                ins.push(In::one(at));
                given.push(Given::One(v[0]));
            }
            Input::Column => {
                let (at, v) = up(n);
                ins.push(In::column(at));
                given.push(Given::Column(v));
            }
            Input::Row => {
                let (at, v) = up(m);
                ins.push(In::row(at));
                given.push(Given::Row(v));
            }
            Input::Tile => {
                let (at, v) = up(m * n);
                ins.push(In::tile(at, n as u32));
                given.push(Given::Tile(v));
            }
            Input::Head(size) => {
                let (at, v) = up(size as usize);
                ins.push(In::head(at, size));
                given.push(Given::Column(v));
            }
            // the contraction's own: no plane, nothing given
            Input::Norm => ins.push(In::norm()),
        }
    }
    (ins, given)
}

/// Each skeleton at the rows it serves, each epilogue filled in, against the
/// interpreter.
#[test]
fn each_skeleton_lands_each_epilogue_as_the_interpreter() {
    // (skeleton, rows, n, k): the gemv split over K and not (past
    // `kernel.gemv.splitk_max_n` columns), the few rows at one, a ragged
    // few and sixteen, the skinny at a ragged few and a full tile.
    #[rustfmt::skip]
    let shapes = [
        (Skeleton::Gemv, 1, 256, 512), (Skeleton::Gemv, 1, 4224, 256),
        (Skeleton::Rows, 1, 192, 512), (Skeleton::Rows, 5, 320, 1024), (Skeleton::Rows, 16, 128, 256),
        (Skeleton::Skinny, 3, 128, 256), (Skeleton::Skinny, 64, 256, 512),
    ];
    let tuning = Tuning::DEFAULT;
    for (skeleton, m, n, k) in shapes {
        for case in Case::ALL {
            let anchored = case.anchored(n, k);
            let epi = Epilogue::interned(&anchored).expect("an epilogue the skeletons take");
            let what = format!("{skeleton:?} {case:?} {m}x{n}x{k}");
            if !skeleton.covers(epi, m as i32, n as i32, k as i32, &tuning) {
                assert_eq!(
                    (skeleton, case),
                    (Skeleton::Rows, Case::Geglu),
                    "{what} is covered"
                );
                continue;
            }
            let mut gpu = Gpu::open();
            let mut lcg = Lcg::seeded((m * 31 + n * 7 + k) as u64);
            let parts = anchored.contraction.layout.parts() as usize;
            let (wr, wv) = lcg.row(parts * n * k);
            let (ar, av) = lcg.row(m * k);
            let (w, act) = (gpu.up(&wr), gpu.up(&ar));
            let (ins, given) = inputs(&mut gpu, &mut lcg, case, m, n);
            let outs: Vec<u64> = (0..epi.results()).map(|_| gpu.zeros(m * n * 2)).collect();
            let ctx = gpu.ctx();
            let fire = || {
                skeleton
                    .fire(&ctx, epi, w, act, &ins, &outs, m as i32, n as i32, k as i32)
                    .unwrap_or_else(|e| panic!("{what} fires: {e:?}"));
            };
            fire();
            gpu.sync();
            let want = eval(&anchored, m, &av, &wv, &[], &given).expect("the interpreter runs");
            let got: Vec<Vec<u16>> = outs.iter().map(|&at| gpu.down(at, m * n)).collect();
            for (r, (got, want)) in got.iter().zip(&want).enumerate() {
                let want: Vec<f32> = want.iter().map(|&v| round_bf16(v)).collect();
                assert_within(&format!("{what}: result {r}"), got, &want, case.tolerance());
            }
            // The products sum in a fixed order: a second fire lands the
            // same bits.
            fire();
            gpu.sync();
            for (r, (again, got)) in outs.iter().zip(&got).enumerate() {
                let again: Vec<u16> = gpu.down(*again, m * n);
                deterministic(&format!("{what}: result {r}"), got, &again);
            }
        }
    }
}

/// A soft cap's and GEGLU's epilogues through the tuner's anchored entry,
/// its fused skeletons timed against the product then the region on the row
/// skeleton: whichever it picks lands the interpreter's result. A shape no
/// skeleton covers lands it on the unfused leg, untimed, a vector a row read
/// one value a row; one whose leg the row skeleton does not carry either (a row
/// in no whole runs of eight) is left to the caller, and `skeleton::serves`
/// says so before anything fires.
#[test]
fn the_anchored_entry_lands_an_epilogue_on_its_pick() {
    // Each over its products rounded to bf16 and over them as they are
    // summed (the compiler's anchors: no op between the product and its
    // epilogue rounds it), the unfused leg then landing them f32.
    for (case, m, n, k, summed) in [
        (Case::Softcap, 4, 512, 256, false),
        (Case::Softcap, 1, 9216, 256, false),
        (Case::Geglu, 8, 256, 512, false),
        (Case::Softcap, 4, 512, 256, true),
        (Case::Geglu, 8, 256, 512, true),
        // Past every skeleton's rows: the unfused leg, in place and in scratch.
        (Case::Softcap, 100, 64, 128, false),
        (Case::Geglu, 100, 64, 128, false),
        (Case::Softcap, 100, 64, 128, true),
        (Case::Geglu, 100, 64, 128, true),
    ] {
        let mut anchored = case.anchored(n, k);
        if summed {
            anchored.contraction.acc = None;
        }
        let epi = Epilogue::interned(&anchored).expect("the skeletons take it");
        assert!(
            skeleton::serves(&anchored, &[], &[Bf16]).is_ok(),
            "{case:?}"
        );
        let parts = anchored.contraction.layout.parts() as usize;
        let mut gpu = Gpu::open();
        let mut lcg = Lcg::seeded((m + n + k) as u64);
        let (wr, wv) = lcg.row(parts * n * k);
        let (ar, av) = lcg.row(m * k);
        let (w, act, y) = (gpu.up(&wr), gpu.up(&ar), gpu.zeros(m * n * 2));
        let ctx = gpu.ctx();
        let (a, wt, yt) = (
            t(act, m, k, Bf16),
            t(w, parts * n, k, Bf16),
            t(y, m, n, Bf16),
        );
        let what = format!("{case:?} {m}x{n}x{k}, summed {summed}");
        // Twice: the first call times the shape's candidates, the second
        // fires the pick.
        for _ in 0..2 {
            let served = gemm::anchored(&ctx, "test.anchored", epi, a, wt, &[], &[yt])
                .unwrap_or_else(|e| panic!("{what} fires: {e:?}"));
            assert!(served, "{what} is served");
        }
        gpu.sync();
        let want = eval(&anchored, m, &av, &wv, &[], &[]).expect("runs");
        let want: Vec<f32> = want[0].iter().map(|&v| round_bf16(v)).collect();
        assert_within(&what, &gpu.down::<u16>(y, m * n), &want, case.tolerance());
    }
    // A row's scale past every skeleton's rows: the leg reads it one value a
    // row.
    let (m, n, k) = (100, 64, 128);
    let row_scaled = |n: usize| Anchored {
        epilogue: Some(Pointwise {
            region: Region {
                width: n as u32,
                params: vec![Shape::Row, Shape::Scalar],
                ops: vec![Op::Binary(Binary::Mul, 0, 1)],
                results: vec![2],
                rounds: vec![(2, Bf16)],
            },
            inputs: vec![Input::Row],
        }),
        ..Anchored::plain(contraction(n, k, Layout::Dense, Some(Bf16)))
    };
    let anchored = row_scaled(n);
    assert!(skeleton::serves(&anchored, &[Bf16], &[Bf16]).is_ok());
    let epi = Epilogue::interned(&anchored).expect("the skeletons take it");
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(41);
    let ((wr, wv), (ar, av), (sr, sv)) = (lcg.row(n * k), lcg.row(m * k), lcg.row(m));
    let (w, act, s, y) = (gpu.up(&wr), gpu.up(&ar), gpu.up(&sr), gpu.zeros(m * n * 2));
    let (a, wt, yt) = (t(act, m, k, Bf16), t(w, n, k, Bf16), t(y, m, n, Bf16));
    let served = gemm::anchored(
        &gpu.ctx(),
        "test.anchored",
        epi,
        a,
        wt,
        &[In::row(s)],
        &[yt],
    );
    assert_eq!(served.ok(), Some(true), "the leg serves {m} rows");
    gpu.sync();
    let want = eval(&anchored, m, &av, &wv, &[], &[Given::Row(sv)]).expect("runs");
    let want: Vec<f32> = want[0].iter().map(|&v| round_bf16(v)).collect();
    assert_within("a row's scale", &gpu.down::<u16>(y, m * n), &want, ONE_MORE);
    // A row in no whole runs of eight: no skeleton past a few rows, nor a leg.
    let narrow = row_scaled(4);
    assert!(skeleton::serves(&narrow, &[Bf16], &[Bf16]).is_err());
    let epi = Epilogue::interned(&narrow).expect("the skeletons take it");
    let y = gpu.zeros(m * 4 * 2);
    let served = gemm::anchored(
        &gpu.ctx(),
        "test.anchored",
        epi,
        t(act, m, k, Bf16),
        t(w, 4, k, Bf16),
        &[In::row(s)],
        &[t(y, m, 4, Bf16)],
    );
    assert_eq!(
        served.ok(),
        Some(false),
        "no skeleton covers {m} rows, nor a leg"
    );
}

/// An epilogue cuBLASLt lands itself (a bias, a tanh GELU, a ReLU, or a
/// bias then either), written as the compiler lowers it.
fn lt_region(n: usize, gelu: bool, relu: bool, biased: bool) -> (Region, Vec<Input>) {
    let mut params = vec![Shape::Row];
    let mut ops = Vec::new();
    let mut rounds = Vec::new();
    let mut x = 0u32;
    if biased {
        params.push(Shape::Row);
        ops.push(Op::Binary(Binary::Add, 0, 1)); // 2
        x = 2;
        rounds.push((2, Bf16));
    }
    let at = |ops: &Vec<Op>| params.len() as u32 + ops.len() as u32;
    if gelu {
        let half = at(&ops);
        ops.push(Op::Const(Literal::F32(0.5)));
        ops.push(Op::Const(Literal::F32(1.0)));
        ops.push(Op::Const(Literal::F32(0.797_884_6)));
        ops.push(Op::Const(Literal::F32(0.044_715)));
        let (one, c, cubic) = (half + 1, half + 2, half + 3);
        let t2 = at(&ops);
        ops.push(Op::Binary(Binary::Mul, x, x));
        ops.push(Op::Binary(Binary::Mul, t2, x));
        ops.push(Op::Binary(Binary::Mul, cubic, t2 + 1));
        ops.push(Op::Binary(Binary::Add, x, t2 + 2));
        ops.push(Op::Binary(Binary::Mul, c, t2 + 3));
        ops.push(Op::Unary(Unary::Tanh, t2 + 4));
        ops.push(Op::Binary(Binary::Add, one, t2 + 5));
        ops.push(Op::Binary(Binary::Mul, half, x));
        ops.push(Op::Binary(Binary::Mul, t2 + 7, t2 + 6));
        x = t2 + 8;
    } else if relu {
        let zero = at(&ops);
        ops.push(Op::Const(Literal::F32(0.0)));
        ops.push(Op::Binary(Binary::Max, x, zero));
        x = zero + 1;
    }
    if !rounds.iter().any(|(v, _)| *v == x) {
        rounds.push((x, Bf16));
    }
    let inputs = if biased { vec![Input::Column] } else { vec![] };
    (
        Region {
            width: n as u32,
            params,
            ops,
            results: vec![x],
            rounds,
        },
        inputs,
    )
}

/// cuBLASLt's own epilogue, the fused candidate the tuner times beside the
/// skeletons and the unfused leg, lands what the anchored region states over
/// its products as they are summed: a bias, a ReLU and a bias then a ReLU,
/// at a decode's rows and a prefill's, within the product's sum in another
/// order (an ulp, two over a long cancelling K; the epilogue's one rounding
/// is the IR's), and
/// GLM-4.7's biased banks (q, q and k and v, k and v) so. A region no
/// cuBLASLt epilogue lands (a soft cap, a tanh GELU, a bias then one, a bias
/// over products rounded first) is not offered to it. Through the tuner's
/// entry a prefill's lands the same on whichever it picks.
#[test]
fn cublaslts_own_epilogue_lands_the_anchored_region() {
    for gelu_biased in [false, true] {
        let (region, inputs) = lt_region(64, true, false, gelu_biased);
        let anchored = Anchored {
            epilogue: Some(Pointwise { region, inputs }),
            ..Anchored::plain(contraction(64, 128, Layout::Dense, None))
        };
        let epi = Epilogue::interned(&anchored).expect("the skeletons take it");
        assert!(epi.lt().is_none(), "a GELU is cuBLASLt's own approximation");
    }
    // A bias over products rounded first is not cuBLASLt's one sum.
    let (region, inputs) = lt_region(64, false, false, true);
    let rounded = Anchored {
        epilogue: Some(Pointwise { region, inputs }),
        ..Anchored::plain(contraction(64, 128, Layout::Dense, Some(Bf16)))
    };
    let epi = Epilogue::interned(&rounded).expect("the skeletons take it");
    assert!(
        epi.lt().is_none(),
        "cuBLASLt rounds no product before its bias"
    );
    for (n, k, relu, biased, rows) in [
        (320usize, 256usize, false, true, &[1usize, 37, 300][..]),
        (320, 256, true, false, &[1, 37, 300]),
        (320, 256, true, true, &[1, 37, 300]),
        (14_336, 5120, false, true, &[1, 2, 100]),
        (12_288, 5120, false, true, &[1, 2, 4, 128]),
        (1024, 5120, false, true, &[16, 128, 256]),
    ] {
        let gelu = false;
        let (region, inputs) = lt_region(n, gelu, relu, biased);
        let anchored = Anchored {
            epilogue: Some(Pointwise { region, inputs }),
            ..Anchored::plain(contraction(n, k, Layout::Dense, None))
        };
        let epi = Epilogue::interned(&anchored).expect("the skeletons take it");
        assert!(epi.lt().is_some(), "relu {relu} bias {biased}");
        for &m in rows {
            let what = format!("relu {relu} bias {biased}, {m}x{n}x{k}");
            let mut gpu = Gpu::open();
            let mut lcg = Lcg::seeded((m * 7 + n) as u64);
            let (wr, wv) = lcg.row(n * k);
            let (ar, av) = lcg.row(m * k);
            let (br, bv) = lcg.row(n);
            let (w, act, b, y) = (gpu.up(&wr), gpu.up(&ar), gpu.up(&br), gpu.zeros(m * n * 2));
            let ins: Vec<In> = if biased { vec![In::column(b)] } else { vec![] };
            let given: Vec<Given> = if biased {
                vec![Given::Column(bv.clone())]
            } else {
                vec![]
            };
            let ctx = gpu.ctx();
            let landed = gemm::anchored_on_lt(
                &ctx,
                "test.lt",
                epi,
                t(act, m, k, Bf16),
                t(w, n, k, Bf16),
                &ins,
                t(y, m, n, Bf16),
            )
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert!(landed, "{what}: some heuristic takes the epilogue");
            gpu.sync();
            let want = eval(&anchored, m, &av, &wv, &[], &given).expect("runs");
            let want: Vec<f32> = want[0].iter().map(|&v| round_bf16(v)).collect();
            assert_within(&what, &gpu.down::<u16>(y, m * n), &want, ONE_MORE);
            // And through the tuner, a prefill's rows past every skeleton.
            if m >= 100 {
                let y2 = gpu.zeros(m * n * 2);
                for _ in 0..2 {
                    let served = gemm::anchored(
                        &gpu.ctx(),
                        "test.lt",
                        epi,
                        t(act, m, k, Bf16),
                        t(w, n, k, Bf16),
                        &ins,
                        &[t(y2, m, n, Bf16)],
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    assert!(served, "{what}");
                }
                gpu.sync();
                assert_within(
                    &format!("{what}, on the tuner's pick"),
                    &gpu.down::<u16>(y2, m * n),
                    &want,
                    ONE_MORE,
                );
            }
        }
    }
    assert!(
        Epilogue::interned(&Case::Softcap.anchored(64, 128))
            .expect("a cap")
            .lt()
            .is_none()
    );
}

/// A packed qkv product's attention prologue as the compiler lowers it: each
/// head normed by its own mean square (q's and k's heads each by their own
/// weight, one head wide, where `normed`), q's and k's heads turned at the
/// row's position (NeoX halves, every column of the head), v's as they are,
/// each landing its own window: q, k, v.
fn qkv(q_heads: usize, kv_heads: usize, hd: usize, k: usize, normed: bool) -> Anchored {
    use kernels_ir::anchor::Window;
    use kernels_ir::build::{Ops, Pairs, Turn};
    let (q, kv) = (q_heads * hd, kv_heads * hd);
    let n = q + 2 * kv;
    let params = if normed { 4 } else { 2 };
    let mut b = Ops::new(params);
    let pos = params - 1;
    let x = if normed {
        let normed = b.rms_normed_heads(0, hd as u32, 1e-6);
        let col = b.push(Op::Column);
        let edge = b.u32(q as u32);
        let in_q = b.push(Op::Binary(Binary::Lt, col, edge));
        let w = b.push(Op::Select(in_q, 1, 2));
        b.push(Op::Binary(Binary::Mul, normed, w))
    } else {
        0
    };
    let turn = Turn {
        head_dim: hd as u32,
        pairs: Pairs::Halves {
            rotary: hd as u32,
            span: hd as u32 / 2,
        },
        theta: 1_000_000.0,
        inverse: false,
        ramp: None,
    };
    let turned = b.rotate(x, pos, &turn);
    let mut rounds = vec![(turned, Bf16)];
    if normed {
        rounds.push((x, Bf16));
    }
    let mut inputs = Vec::new();
    if normed {
        inputs.extend([Input::Head(hd as u32), Input::Head(hd as u32)]);
    }
    inputs.push(Input::Row);
    let mut shapes = vec![Shape::Row; params as usize - 1];
    shapes.push(Shape::Scalar);
    Anchored {
        epilogue: Some(Pointwise {
            region: Region {
                width: n as u32,
                params: shapes,
                ops: b.ops,
                results: vec![turned, turned, 0],
                rounds,
            },
            inputs,
        }),
        windows: vec![
            Window {
                from: 0,
                width: q as u32,
            },
            Window {
                from: q as u32,
                width: kv as u32,
            },
            Window {
                from: (q + kv) as u32,
                width: kv as u32,
            },
        ],
        ..Anchored::plain(contraction(n, k, Layout::Dense, None))
    }
}

/// A product's ulp, then a norm's and a turn's ([`crate::elemwise::within`]'s
/// `TURN`) over it.
const TURNED: f32 = PRODUCT + 4.0;

/// The inputs of [`qkv`]'s region at `m` rows: the heads' weights (where
/// normed) and the rows' positions, stored i32.
fn qkv_inputs(
    gpu: &mut Gpu,
    lcg: &mut Lcg,
    m: usize,
    hd: usize,
    normed: bool,
) -> (Vec<In>, Vec<Given>, Vec<Dtype>) {
    let (mut ins, mut given, mut stored) = (Vec::new(), Vec::new(), Vec::new());
    if normed {
        for _ in 0..2 {
            let (raw, exact) = lcg.row(hd);
            ins.push(In::head(gpu.up(&raw), hd as u32));
            given.push(Given::Column(exact));
            stored.push(Bf16);
        }
    }
    let pos: Vec<i32> = (0..m).map(|r| (r * 977 % 4093) as i32).collect();
    ins.push(In::row(gpu.up(&pos)).i32());
    given.push(Given::Row(pos.iter().map(|&p| p as f32).collect()));
    stored.push(Dtype::I32);
    (ins, given, stored)
}

/// [`qkv`]'s epilogue lands each window as the interpreter does: on the
/// skinny skeleton, whose block holds whole heads (64, 128 and 256 wide, a
/// head's lanes in one warp), at a decode's rows and a full tile, and
/// through the tuner's entry, a prefill's rows past the skinny's on the
/// unfused leg (one piece a window, each its columns shifted). The gemv and
/// the few-rows skeletons, a column a warp, hold no head.
#[test]
fn a_heads_norm_and_turn_lands_its_windows_as_the_interpreter() {
    let tuning = Tuning::DEFAULT;
    #[rustfmt::skip]
    let shapes = [
        (1, 4, 2, 64, 256, true), (5, 4, 1, 128, 512, true), (64, 2, 1, 256, 256, true),
        (3, 8, 2, 128, 256, false), (1, 16, 8, 128, 1024, true),
    ];
    for (m, qh, kvh, hd, k, normed) in shapes {
        let anchored = qkv(qh, kvh, hd, k, normed);
        assert_eq!(anchored.verify(), Ok(()));
        let n = (qh + 2 * kvh) * hd;
        let what = format!("{qh}+{kvh}x{hd} heads, {m}x{k}, normed {normed}");
        let mut gpu = Gpu::open();
        let mut lcg = Lcg::seeded((m * 13 + n + k) as u64);
        let (ins, given, stored) = qkv_inputs(&mut gpu, &mut lcg, 100, hd, normed);
        let epi = Epilogue::interned_stored(&anchored, &stored).expect("the skeletons take it");
        assert_eq!(epi.span(), hd as u32, "{what}");
        assert!(
            skeleton::serves(&anchored, &stored, &[Bf16; 3]).is_ok(),
            "{what}"
        );
        for other in [Skeleton::Gemv, Skeleton::Rows] {
            assert!(!other.covers(epi, m as i32, n as i32, k as i32, &tuning));
        }
        assert!(
            Skeleton::Skinny.covers(epi, m as i32, n as i32, k as i32, &tuning),
            "{what}"
        );
        let widths = [qh * hd, kvh * hd, kvh * hd];
        for (rows, through) in [(m, false), (m, true), (100, true)] {
            let (wr, wv) = lcg.row(n * k);
            let (ar, av) = lcg.row(rows * k);
            let (w, act) = (gpu.up(&wr), gpu.up(&ar));
            let outs: Vec<u64> = widths.iter().map(|wd| gpu.zeros(rows * wd * 2)).collect();
            let fire = |gpu: &Gpu| {
                let ctx = gpu.ctx();
                if through {
                    let landed: Vec<_> = outs
                        .iter()
                        .zip(widths)
                        .map(|(&o, wd)| t(o, rows, wd, Bf16))
                        .collect();
                    let served = gemm::anchored(
                        &ctx,
                        "test.qkv",
                        epi,
                        t(act, rows, k, Bf16),
                        t(w, n, k, Bf16),
                        &ins,
                        &landed,
                    )
                    .unwrap_or_else(|e| panic!("{what} fires: {e:?}"));
                    assert!(served, "{what}: served at {rows} rows");
                } else {
                    Skeleton::Skinny
                        .fire(
                            &ctx,
                            epi,
                            w,
                            act,
                            &ins,
                            &outs,
                            rows as i32,
                            n as i32,
                            k as i32,
                        )
                        .unwrap_or_else(|e| panic!("{what} fires: {e:?}"));
                }
            };
            // The tuner's first call times its candidates, the second fires
            // its pick.
            fire(&gpu);
            fire(&gpu);
            gpu.sync();
            let given: Vec<Given> = given
                .iter()
                .map(|g| match g {
                    Given::Row(p) => Given::Row(p[..rows].to_vec()),
                    other => other.clone(),
                })
                .collect();
            let want = eval(&anchored, rows, &av, &wv, &[], &given).expect("the interpreter runs");
            let mut got = Vec::new();
            for (r, (&at, wd)) in outs.iter().zip(widths).enumerate() {
                let landed: Vec<u16> = gpu.down(at, rows * wd);
                let want: Vec<f32> = want[r].iter().map(|&v| round_bf16(v)).collect();
                let case = format!("{what}: {rows} rows through the tuner {through}, window {r}");
                assert_within(&case, &landed, &want, TURNED);
                got.push(landed);
            }
            fire(&gpu);
            gpu.sync();
            for (r, (&at, wd)) in outs.iter().zip(widths).enumerate() {
                let again: Vec<u16> = gpu.down(at, rows * wd);
                deterministic(&format!("{what}: window {r}"), &got[r], &again);
            }
        }
    }
}

/// A turn of every head of a row wider than the row skeleton holds (q
/// alone, no windows): its unfused leg cut into slices of whole heads, each
/// reading its columns shifted to the slice's, lands what the whole row
/// does.
#[test]
fn a_turned_row_past_the_row_skeletons_width_lands_on_its_slices() {
    let (hd, heads, k, m) = (128usize, 72usize, 256usize, 100usize);
    let n = hd * heads;
    let mut anchored = qkv(heads, 0, hd, k, false);
    anchored.windows.clear();
    if let Some(epi) = anchored.epilogue.as_mut() {
        epi.region.results = vec![epi.region.results[0]];
    }
    anchored.contraction.n = n as u32;
    if let Some(epi) = anchored.epilogue.as_mut() {
        epi.region.width = n as u32;
    }
    assert_eq!(anchored.verify(), Ok(()));
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(77);
    let (ins, given, stored) = qkv_inputs(&mut gpu, &mut lcg, m, hd, false);
    let epi = Epilogue::interned_stored(&anchored, &stored).expect("the skeletons take it");
    assert!(skeleton::serves(&anchored, &stored, &[Bf16]).is_ok());
    let (wr, wv) = lcg.row(n * k);
    let (ar, av) = lcg.row(m * k);
    let (w, act, y) = (gpu.up(&wr), gpu.up(&ar), gpu.zeros(m * n * 2));
    for _ in 0..2 {
        let served = gemm::anchored(
            &gpu.ctx(),
            "test.turn",
            epi,
            t(act, m, k, Bf16),
            t(w, n, k, Bf16),
            &ins,
            &[t(y, m, n, Bf16)],
        )
        .unwrap_or_else(|e| panic!("{e:?}"));
        assert!(served);
    }
    gpu.sync();
    let want = eval(&anchored, m, &av, &wv, &[], &given).expect("runs");
    let want: Vec<f32> = want[0].iter().map(|&v| round_bf16(v)).collect();
    assert_within("a turn's slices", &gpu.down::<u16>(y, m * n), &want, TURNED);
}

/// `rmsnorm(a, g) · wᵀ`, then a residual, as the rewrite states it: the
/// bank the gain folded into (`fold_gain` on the device landing the bits of
/// the declared `fold_bank`), each row's norm scale the skeleton's own as it
/// streams the row, the epilogue `y · r + res`. Each skeleton at the rows it
/// serves, and the tuner's entry at a prefill's rows past them (the unfused
/// leg, its scales by `norm_scales`), land what the interpreter does.
#[test]
fn a_normed_operand_lands_its_scaled_products_as_the_interpreter() {
    use kernels_cuda::linear::cast::fold_gain;
    use kernels_cuda::linear::skeleton::{Gain, RowNorm};
    use kernels_ir::anchor::fold_bank;
    let anchored = |n: usize, k: usize, gain: Gain| Anchored {
        norm: Some(RowNorm { eps: 1e-6, gain }),
        epilogue: Some(Pointwise {
            region: Region {
                width: n as u32,
                params: vec![Shape::Row, Shape::Scalar, Shape::Row],
                ops: vec![
                    Op::Binary(Binary::Mul, 0, 1), // 3
                    Op::Binary(Binary::Add, 3, 2), // 4
                ],
                results: vec![4],
                rounds: vec![(4, Bf16)],
            },
            inputs: vec![Input::Norm, Input::Tile],
        }),
        ..Anchored::plain(contraction(n, k, Layout::Dense, None))
    };
    let tuning = Tuning::DEFAULT;
    #[rustfmt::skip]
    let shapes = [
        (Skeleton::Gemv, 1, 256, 512), (Skeleton::Gemv, 1, 4224, 256),
        (Skeleton::Rows, 1, 192, 512), (Skeleton::Rows, 5, 320, 1024), (Skeleton::Rows, 16, 128, 256),
        (Skeleton::Skinny, 3, 128, 256), (Skeleton::Skinny, 64, 256, 512),
    ];
    for (skeleton, m, n, k) in shapes {
        for gain in [Gain::Scale, Gain::PlusOne] {
            let a = anchored(n, k, gain);
            assert_eq!(a.verify(), Ok(()));
            let epi = Epilogue::interned(&a).expect("the skeletons take it");
            assert!(skeleton.covers(epi, m as i32, n as i32, k as i32, &tuning));
            let what = format!("{skeleton:?} {gain:?} {m}x{n}x{k}");
            let mut gpu = Gpu::open();
            let mut lcg = Lcg::seeded((m * 17 + n + k) as u64);
            let ((wr, wv), (ar, av), (gr, gv), (rr, rv)) =
                (lcg.row(n * k), lcg.row(m * k), lcg.row(k), lcg.row(m * n));
            let (w, act, g, res) = (gpu.up(&wr), gpu.up(&ar), gpu.up(&gr), gpu.up(&rr));
            let ctx = gpu.ctx();
            let mut bank = t(w, n, k, Bf16);
            fold_gain(&ctx, &mut bank, t(g, 1, k, Bf16), gain == Gain::PlusOne)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            gpu.sync();
            let folded = fold_bank(&a.norm.expect("a norm"), &wv, &gv);
            let landed: Vec<u16> = gpu.down(w, n * k);
            let bits: Vec<u16> = folded.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            assert!(landed == bits, "{what}: the device's fold is the declared one");
            let y = gpu.zeros(m * n * 2);
            let ins = [In::norm(), In::tile(res, n as u32)];
            skeleton
                .fire(&ctx, epi, w, act, &ins, &[y], m as i32, n as i32, k as i32)
                .unwrap_or_else(|e| panic!("{what} fires: {e:?}"));
            gpu.sync();
            let want = eval(&a, m, &av, &folded, &[], &[Given::Tile(rv.clone())]).expect("runs");
            let want: Vec<f32> = want[0].iter().map(|&v| round_bf16(v)).collect();
            assert_within(&what, &gpu.down::<u16>(y, m * n), &want, ONE_MORE);
            // Past every skeleton's rows, through the tuner: the leg.
            if skeleton == Skeleton::Skinny && m == 64 {
                let rows = 100;
                let ((ar, av), (rr, rv)) = (lcg.row(rows * k), lcg.row(rows * n));
                let (act, res, y) = (gpu.up(&ar), gpu.up(&rr), gpu.zeros(rows * n * 2));
                let ins = [In::norm(), In::tile(res, n as u32)];
                for _ in 0..2 {
                    let served = gemm::anchored(
                        &gpu.ctx(),
                        "test.normed",
                        epi,
                        t(act, rows, k, Bf16),
                        t(w, n, k, Bf16),
                        &ins,
                        &[t(y, rows, n, Bf16)],
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    assert!(served);
                }
                gpu.sync();
                let want = eval(&a, rows, &av, &folded, &[], &[Given::Tile(rv)]).expect("runs");
                let want: Vec<f32> = want[0].iter().map(|&v| round_bf16(v)).collect();
                assert_within(
                    &format!("{what}, {rows} rows on the leg"),
                    &gpu.down::<u16>(y, rows * n),
                    &want,
                    ONE_MORE,
                );
            }
        }
    }
}

/// `anchored` with its operand's rows normed: the product scaled by each
/// row's norm scale (an input appended, the contraction's own) before the
/// epilogue reads it, every read of the product reading the scaled one.
fn normed(mut anchored: Anchored) -> Anchored {
    use kernels_cuda::linear::skeleton::{Gain, RowNorm};
    let epi = anchored.epilogue.as_mut().expect("an epilogue");
    let region = &mut epi.region;
    let params = region.params.len() as u32;
    let scaled = params + 1;
    let at = |v: u32| match v {
        0 => scaled,
        v if v < params => v,
        v => v + 2,
    };
    let mut ops = vec![Op::Binary(Binary::Mul, 0, params)];
    for op in &region.ops {
        ops.push(match *op {
            Op::Unary(u, a) => Op::Unary(u, at(a)),
            Op::Binary(b, x, y) => Op::Binary(b, at(x), at(y)),
            Op::Select(c, x, y) => Op::Select(at(c), at(x), at(y)),
            Op::Cast(a, e) => Op::Cast(at(a), e),
            Op::Broadcast(a) => Op::Broadcast(at(a)),
            Op::Reduce(r, a) => Op::Reduce(r, at(a)),
            Op::ReduceHeads(r, a, s) => Op::ReduceHeads(r, at(a), s),
            Op::Fma(x, y, z) => Op::Fma(at(x), at(y), at(z)),
            Op::Peer(s, c) => Op::Peer(at(s), at(c)),
            other => other,
        });
    }
    region.ops = ops;
    region.params.push(Shape::Scalar);
    region.results = region.results.iter().map(|&r| at(r)).collect();
    region.rounds = region.rounds.iter().map(|&(v, d)| (at(v), d)).collect();
    epi.inputs.push(Input::Norm);
    anchored.norm = Some(RowNorm {
        eps: 1e-6,
        gain: Gain::Scale,
    });
    anchored
}

/// A normed operand under a head's epilogue (a packed qkv's split, norms and
/// turn over its scaled products): on the skinny skeleton at its rows and
/// the leg's past them, as the interpreter lands it.
#[test]
fn a_normed_operands_heads_land_their_windows_as_the_interpreter() {
    let (qh, kvh, hd, k) = (4usize, 2usize, 128usize, 512usize);
    let anchored = normed(qkv(qh, kvh, hd, k, true));
    assert_eq!(anchored.verify(), Ok(()));
    let n = (qh + 2 * kvh) * hd;
    let widths = [qh * hd, kvh * hd, kvh * hd];
    for rows in [1usize, 5, 64, 100] {
        let mut gpu = Gpu::open();
        let mut lcg = Lcg::seeded(rows as u64 + 3);
        let (mut ins, given, stored) = qkv_inputs(&mut gpu, &mut lcg, rows, hd, true);
        ins.push(In::norm());
        let epi = Epilogue::interned_stored(&anchored, &stored).expect("the skeletons take it");
        let ((wr, wv), (ar, av)) = (lcg.row(n * k), lcg.row(rows * k));
        let (w, act) = (gpu.up(&wr), gpu.up(&ar));
        let outs: Vec<u64> = widths.iter().map(|wd| gpu.zeros(rows * wd * 2)).collect();
        let landed: Vec<_> = outs
            .iter()
            .zip(widths)
            .map(|(&o, wd)| t(o, rows, wd, Bf16))
            .collect();
        for _ in 0..2 {
            let served = gemm::anchored(
                &gpu.ctx(),
                "test.normed_qkv",
                epi,
                t(act, rows, k, Bf16),
                t(w, n, k, Bf16),
                &ins,
                &landed,
            )
            .unwrap_or_else(|e| panic!("{rows} rows: {e:?}"));
            assert!(served);
        }
        gpu.sync();
        let want = eval(&anchored, rows, &av, &wv, &[], &given).expect("runs");
        for (r, (&at, wd)) in outs.iter().zip(widths).enumerate() {
            let want: Vec<f32> = want[r].iter().map(|&v| round_bf16(v)).collect();
            assert_within(
                &format!("{rows} rows, window {r}"),
                &gpu.down::<u16>(at, rows * wd),
                &want,
                TURNED,
            );
        }
    }
}
