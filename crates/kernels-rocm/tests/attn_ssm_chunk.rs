//! The chunk-parallel gated delta prefill against the serial scan it
//! replaces: requests spanning several chunks, one cut at an interior begin
//! and capped by a commit.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, STATE, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::ssm;
use kernels_rocm::tuning::Tuning;
use kernels_rocm::{RaggedTensor, RecurrentPool, Tensor};

struct Case {
    lens: Vec<usize>,
    k_heads: u32,
    v_heads: u32,
    begin: Option<Vec<i32>>,
    commit: Option<Vec<i32>>,
}

const D: u32 = 128;

/// The fires share the context's named scratch planes, so the tests in this
/// file take turns on the device.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Inputs {
    qkv: Vec<f32>,
    gates: Vec<f32>,
    slab: Vec<f32>,
    indptr: Vec<i32>,
    slots: Vec<i32>,
}

impl Case {
    fn rows(&self) -> usize {
        self.lens.iter().sum()
    }
    fn conv_dim(&self) -> u32 {
        2 * self.k_heads * D + self.v_heads * D
    }
    fn stride(&self) -> usize {
        (self.v_heads * D * D) as usize
    }
    fn inputs(&self, seed: u64) -> Inputs {
        let (rows, vh) = (self.rows(), self.v_heads as usize);
        let qkv = values(rows * self.conv_dim() as usize, seed, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let mut gates = Vec::with_capacity(rows * 2 * vh);
        for (i, v) in values(rows * 2 * vh, seed + 1, 1.0).into_iter().enumerate() {
            gates.push(if i % (2 * vh) < vh {
                -0.02 - 0.2 * v.abs()
            } else {
                0.5 + 0.45 * v
            });
        }
        let lanes = self.lens.len();
        let slab = values(lanes * self.stride(), seed + 2, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect();
        let mut indptr = vec![0i32];
        for &l in &self.lens {
            indptr.push(indptr.last().unwrap() + l as i32);
        }
        let slots = (0..lanes as i32).rev().collect();
        Inputs {
            qkv,
            gates,
            slab,
            indptr,
            slots,
        }
    }
}

struct Fired {
    y: Tensor,
    slab: Tensor,
}

struct Bound {
    state: RecurrentPool,
    qkv: RaggedTensor,
    gates: Tensor,
    y: Tensor,
}

fn bind(gpu: &Gpu, case: &Case, inp: &Inputs) -> Bound {
    let rows = case.rows() as u32;
    let lanes = case.lens.len() as u32;
    let slab = gpu.tensor(&inp.slab, lanes, case.stride() as u32, Dtype::Bf16);
    let mut state = RecurrentPool {
        slab,
        slot_ids: gpu.column(&inp.slots),
        slot_stride_elems: case.stride() as i64,
        ..STATE
    };
    if let Some(b) = &case.begin {
        state.begin_at = gpu.column(b);
    }
    if let Some(c) = &case.commit {
        state.commit_len = gpu.column(c);
    }
    let qkv = RaggedTensor {
        data: gpu.tensor(&inp.qkv, rows, case.conv_dim(), Dtype::Bf16),
        indptr: gpu.column(&inp.indptr),
    };
    let gates = gpu.tensor(&inp.gates, rows, 2 * case.v_heads, Dtype::F32);
    let y = gpu.zeros(rows, case.v_heads * D, Dtype::F32);
    Bound {
        state,
        qkv,
        gates,
        y,
    }
}

/// How a fire walks: the serial scan, or the chunked passes at 16- or
/// 32-column state slices.
#[derive(Clone, Copy)]
enum Walk {
    Serial,
    Chunked,
    Wide,
}

fn launch(gpu: &Gpu, case: &Case, b: &mut Bound, walk: Walk) {
    let serial = matches!(walk, Walk::Serial);
    let ctx = gpu.ctx().with_tuning(Tuning {
        ssm_gdn_wide_from: u32::from(matches!(walk, Walk::Wide)),
        ..Tuning::DEFAULT
    });
    let run = if serial {
        ssm::gated_delta_chunked_serial
    } else {
        ssm::gated_delta_chunked
    };
    run(
        &ctx,
        b.qkv,
        Tensor::ABSENT,
        b.gates,
        &b.state,
        case.k_heads,
        case.v_heads,
        D,
        D,
        &mut b.y,
    )
    .unwrap();
}

fn fire(gpu: &Gpu, case: &Case, inp: &Inputs, walk: Walk) -> Fired {
    let mut b = bind(gpu, case, inp);
    launch(gpu, case, &mut b, walk);
    gpu.sync();
    Fired {
        y: b.y,
        slab: b.state.slab,
    }
}

fn worst(have: &[f32], want: &[f32]) -> (f32, usize) {
    let mut w = (0f32, 0usize);
    for (i, (&a, &b)) in have.iter().zip(want).enumerate() {
        let e = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
        if e > w.0 || e.is_nan() {
            w = (if e.is_nan() { f32::INFINITY } else { e }, i);
        }
    }
    w
}

#[test]
fn the_chunked_gated_delta_answers_the_serial_scan() {
    let _turn = TURN.lock().expect("turn lock poisoned");
    let Some(gpu) = Gpu::open() else { return };
    let cases = [
        Case {
            lens: vec![70, 5, 130],
            k_heads: 2,
            v_heads: 4,
            begin: None,
            commit: None,
        },
        Case {
            lens: vec![64, 1, 200, 129],
            k_heads: 4,
            v_heads: 4,
            begin: Some(vec![0, 0, 7, 64]),
            commit: Some(vec![64, 1, 150, 65]),
        },
    ];
    for (n, case) in cases.iter().enumerate() {
        let inp = case.inputs(40 + n as u64);
        let serial = fire(&gpu, case, &inp, Walk::Serial);
        let chunked = fire(&gpu, case, &inp, Walk::Chunked);
        let again = fire(&gpu, case, &inp, Walk::Chunked);
        let wide = fire(&gpu, case, &inp, Walk::Wide);
        let (ys, yc, ya) = (
            gpu.read(&serial.y),
            gpu.read(&chunked.y),
            gpu.read(&again.y),
        );
        let (ss, sc, sa) = (
            gpu.read(&serial.slab),
            gpu.read(&chunked.slab),
            gpu.read(&again.slab),
        );
        assert!(
            yc.iter().zip(&ya).all(|(a, b)| a.to_bits() == b.to_bits()) && sc == sa,
            "case {n}: two chunked fires disagree"
        );
        assert!(
            gpu.read(&wide.y)
                .iter()
                .zip(&yc)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && gpu.read(&wide.slab) == sc,
            "case {n}: the 32-column slices disagree with the 16-column ones"
        );
        let (e, at) = worst(&yc, &ys);
        let width = (case.v_heads * D) as usize;
        assert!(
            e < 3e-2,
            "case {n}: output row {} col {}: serial {} chunked {} ({e})",
            at / width,
            at % width,
            ys[at],
            yc[at]
        );
        let (es, _) = worst(&sc, &ss);
        assert!(es < 3e-2, "case {n}: final states differ by {es}");
        eprintln!("case {n}: worst output error {e}, worst state error {es}");
    }
}

/// A batch's step (a block per head and row, slots reversed) answers what
/// its rows answer stepped one at a time.
#[test]
fn the_batch_step_answers_its_rows_one_at_a_time() {
    let _turn = TURN.lock().expect("turn lock poisoned");
    let Some(gpu) = Gpu::open() else { return };
    let (kh, vh) = (4u32, 16u32);
    let rows = 24u32;
    let c = 2 * kh * D + vh * D;
    let stride = (vh * D * D) as usize;
    let qkv: Vec<f32> = values((rows * c) as usize, 81, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let mut gates = Vec::new();
    for (i, v) in values((rows * 2 * vh) as usize, 82, 1.0)
        .into_iter()
        .enumerate()
    {
        gates.push(if i % (2 * vh as usize) < vh as usize {
            -0.05 - 0.3 * v.abs()
        } else {
            0.5 + 0.45 * v
        });
    }
    let slab: Vec<f32> = values(rows as usize * stride, 83, 0.1)
        .into_iter()
        .map(round_bf16)
        .collect();
    let slots: Vec<i32> = (0..rows as i32).rev().collect();
    let run = |batch: bool| -> (Vec<f32>, Vec<f32>) {
        let slab_t = gpu.tensor(&slab, rows, stride as u32, Dtype::Bf16);
        let mut out = vec![0f32; (rows * vh * D) as usize];
        let pool = |slot_ids: Tensor| RecurrentPool {
            slab: slab_t,
            slot_ids,
            slot_stride_elems: stride as i64,
            ..STATE
        };
        if batch {
            let mut y = gpu.zeros(rows, vh * D, Dtype::F32);
            ssm::gated_delta(
                &gpu.ctx(),
                gpu.tensor(&qkv, rows, c, Dtype::Bf16),
                Tensor::ABSENT,
                gpu.tensor(&gates, rows, 2 * vh, Dtype::F32),
                &pool(gpu.column(&slots)),
                kh,
                vh,
                D,
                D,
                &mut y,
            )
            .unwrap();
            out = gpu.read(&y);
        } else {
            for r in 0..rows as usize {
                let mut y = gpu.zeros(1, vh * D, Dtype::F32);
                let row = &qkv[r * c as usize..(r + 1) * c as usize];
                let gr = &gates[r * 2 * vh as usize..(r + 1) * 2 * vh as usize];
                ssm::gated_delta(
                    &gpu.ctx(),
                    gpu.tensor(row, 1, c, Dtype::Bf16),
                    Tensor::ABSENT,
                    gpu.tensor(gr, 1, 2 * vh, Dtype::F32),
                    &pool(gpu.column(&[slots[r]])),
                    kh,
                    vh,
                    D,
                    D,
                    &mut y,
                )
                .unwrap();
                out[r * (vh * D) as usize..(r + 1) * (vh * D) as usize]
                    .copy_from_slice(&gpu.read(&y));
            }
        }
        gpu.sync();
        (out, gpu.read(&slab_t))
    };
    let (yb, sb) = run(true);
    let (yt, st) = run(false);
    let (e, at) = worst(&yb, &yt);
    assert!(
        e < 2e-3,
        "output {at}: batch {} row by row {} ({e})",
        yb[at],
        yt[at]
    );
    let (es, _) = worst(&sb, &st);
    assert!(es < 1e-2, "states differ by {es}");
    eprintln!("batch vs row-by-row step: output {e}, state {es}");
}
