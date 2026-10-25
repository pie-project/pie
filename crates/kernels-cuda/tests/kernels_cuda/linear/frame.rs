//! The contraction frames (`linear::frame`) with an epilogue filled in land
//! what the anchored region states: each frame and each epilogue against
//! kernels-ir's interpreter (`anchor::eval`), within the tolerances below
//! (in bf16 ulps, as `elemwise::within` counts them), and the same bits on
//! a second fire.
//!
//! - **The contraction's order is the frame's.** The interpreter sums a
//!   product in f64 and rounds it to f32 once; a frame sums it in f32 in
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
use kernels_cuda::linear::frame::{
    self, Anchored, Contraction, Epilogue, Frame, In, Input, Layout, Pointwise, Quant,
};
use kernels_cuda::linear::gemm;
use kernels_cuda::tuning::Tuning;
use kernels_ir::anchor::{Given, eval};
use kernels_ir::frame::round_bf16;
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

/// What a frame is given and what the interpreter is: the epilogue's
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

    /// The anchored region at `n x k`, the region a frame's epilogue is
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

/// A product rounded to bf16, its sum in the frame's order.
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

/// The case's inputs at `m x n`: on the device, as the frame reads them, and
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
        }
    }
    (ins, given)
}

/// Each frame at the rows it serves, each epilogue filled in, against the
/// interpreter.
#[test]
fn each_frame_lands_each_epilogue_as_the_interpreter() {
    // (frame, rows, n, k): the gemv split over K and not (past
    // `kernel.gemv.splitk_max_n` columns), the few rows at one, a ragged
    // few and sixteen, the skinny at a ragged few and a full tile.
    #[rustfmt::skip]
    let shapes = [
        (Frame::Gemv, 1, 256, 512), (Frame::Gemv, 1, 4224, 256),
        (Frame::Rows, 1, 192, 512), (Frame::Rows, 5, 320, 1024), (Frame::Rows, 16, 128, 256),
        (Frame::Skinny, 3, 128, 256), (Frame::Skinny, 64, 256, 512),
    ];
    let tuning = Tuning::DEFAULT;
    for (frame, m, n, k) in shapes {
        for case in Case::ALL {
            let anchored = case.anchored(n, k);
            let epi = Epilogue::interned(&anchored).expect("an epilogue the frames take");
            let what = format!("{frame:?} {case:?} {m}x{n}x{k}");
            if !frame.covers(epi, m as i32, n as i32, k as i32, &tuning) {
                assert_eq!(
                    (frame, case),
                    (Frame::Rows, Case::Geglu),
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
                frame
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
/// its fused frames timed against the product then the region on the row
/// frame: whichever it picks lands the interpreter's result. A shape no
/// frame covers lands it on the unfused leg, untimed, a vector a row read
/// one value a row; one whose leg the row frame does not carry either (a row
/// in no whole runs of eight) is left to the caller, and `frame::serves`
/// says so before anything fires.
#[test]
fn the_anchored_entry_lands_an_epilogue_on_its_pick() {
    for (case, m, n, k) in [
        (Case::Softcap, 4, 512, 256),
        (Case::Softcap, 1, 9216, 256),
        (Case::Geglu, 8, 256, 512),
        // Past every frame's rows: the unfused leg, in place and in scratch.
        (Case::Softcap, 100, 64, 128),
        (Case::Geglu, 100, 64, 128),
    ] {
        let anchored = case.anchored(n, k);
        let epi = Epilogue::interned(&anchored).expect("the frames take it");
        assert!(frame::serves(&anchored, &[], &[Bf16]).is_ok(), "{case:?}");
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
        let what = format!("{case:?} {m}x{n}x{k}");
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
    // A row's scale past every frame's rows: the leg reads it one value a
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
    assert!(frame::serves(&anchored, &[Bf16], &[Bf16]).is_ok());
    let epi = Epilogue::interned(&anchored).expect("the frames take it");
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
    // A row in no whole runs of eight: no frame past a few rows, nor a leg.
    let narrow = row_scaled(4);
    assert!(frame::serves(&narrow, &[Bf16], &[Bf16]).is_err());
    let epi = Epilogue::interned(&narrow).expect("the frames take it");
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
        "no frame covers {m} rows, nor a leg"
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
/// frames and the unfused leg, lands what the anchored region states: a
/// bias, a ReLU and a bias then a ReLU, at a decode's rows and a prefill's,
/// within a product's ulp and the epilogue's rounding; and GLM-4.7's
/// biased banks (q, q and k and v, k and v) at a decode's rows and a
/// prefill's. A region no cuBLASLt epilogue lands (a soft
/// cap, a tanh GELU, a bias then one) is not offered to it. Through the
/// tuner's entry a prefill's lands the same on whichever it picks.
#[test]
fn cublaslts_own_epilogue_lands_the_anchored_region() {
    for gelu_biased in [false, true] {
        let (region, inputs) = lt_region(64, true, false, gelu_biased);
        let anchored = Anchored {
            epilogue: Some(Pointwise { region, inputs }),
            ..Anchored::plain(contraction(64, 128, Layout::Dense, Some(Bf16)))
        };
        let epi = Epilogue::interned(&anchored).expect("the frames take it");
        assert!(epi.lt().is_none(), "a GELU is cuBLASLt's own approximation");
    }
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
            ..Anchored::plain(contraction(n, k, Layout::Dense, Some(Bf16)))
        };
        let epi = Epilogue::interned(&anchored).expect("the frames take it");
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
            // And through the tuner, a prefill's rows past every frame.
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
