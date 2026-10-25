//! The contraction frames (`linear::frame`) with an epilogue filled in land
//! what the anchored region states: each frame and each epilogue against
//! kernels-ir's interpreter (`anchor::eval`), within a bf16 product's
//! tolerance, and the same bits on a second fire. The tuner's anchored
//! entry (`gemm::anchored`) lands the same on whichever candidate it picks.

use crate::common::linear_ref::t;
use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype::{self, Bf16};
use kernels_cuda::linear::frame::{
    Anchored, Contraction, Epilogue, Frame, In, Input, Layout, Pointwise, Quant,
};
use kernels_cuda::linear::gemm;
use kernels_cuda::tuning::Tuning;
use kernels_ir::anchor::{Given, eval};
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

/// The worst distance of `got` from `want`, in units of a bf16 product's
/// tolerance: each element's distance over 1/64 of its value plus the
/// plane's RMS (a near-zero sum's terms are the RMS's size).
fn off(got: &[f32], want: &[f32]) -> f64 {
    let rms = (want.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    got.iter()
        .zip(want)
        .map(|(&g, &w)| {
            let (g, w) = (f64::from(g), f64::from(w));
            (g - w).abs() / ((w.abs() + rms) / 64.0)
        })
        .fold(0.0, f64::max)
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
                let got: Vec<f32> = got.iter().map(|&v| from_bf16(v)).collect();
                let apart = off(&got, want);
                assert!(
                    apart <= 1.0,
                    "{what}: result {r} is {apart:.2} tolerances off"
                );
            }
            // The products sum in a fixed order: a second fire lands the
            // same bits.
            fire();
            gpu.sync();
            let again: Vec<Vec<u16>> = outs.iter().map(|&at| gpu.down(at, m * n)).collect();
            assert_eq!(again, got, "{what} lands again");
        }
    }
}

/// The hand rules' epilogues (`fused.matmul_softcap`'s, `fused.matmul_glu`'s
/// GEGLU) through the tuner's anchored entry, its fused frames timed
/// against the product then the region on the row frame: whichever it
/// picks lands the interpreter's result, and a shape no frame covers is
/// left to the caller.
#[test]
fn the_anchored_entry_lands_the_hand_rules_on_its_pick() {
    for (case, m, n, k) in [
        (Case::Softcap, 4, 512, 256),
        (Case::Softcap, 1, 9216, 256),
        (Case::Geglu, 8, 256, 512),
    ] {
        let epi = match case {
            Case::Softcap => Epilogue::softcap(30.0).expect("a cap"),
            _ => Epilogue::geglu_tanh(),
        };
        let anchored = case.anchored(n, k);
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
        let got: Vec<f32> = gpu
            .down::<u16>(y, m * n)
            .into_iter()
            .map(from_bf16)
            .collect();
        let apart = off(&got, &want[0]);
        assert!(apart <= 1.0, "{what} is {apart:.2} tolerances off");
    }
    // Past every frame's rows: the caller's ops serve it.
    let mut gpu = Gpu::open();
    let (m, n, k) = (100, 64, 128);
    let (w, act, y) = (
        gpu.zeros(n * k * 2),
        gpu.zeros(m * k * 2),
        gpu.zeros(m * n * 2),
    );
    let epi = Epilogue::softcap(30.0).expect("a cap");
    let (a, wt, yt) = (t(act, m, k, Bf16), t(w, n, k, Bf16), t(y, m, n, Bf16));
    let served = gemm::anchored(&gpu.ctx(), "test.anchored", epi, a, wt, &[], &[yt]);
    assert_eq!(served.ok(), Some(false), "no frame covers {m} rows");
}
