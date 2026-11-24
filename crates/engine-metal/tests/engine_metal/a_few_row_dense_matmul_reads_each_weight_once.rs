#![cfg(target_vendor = "apple")]

//! A dense bf16 matmul of 1 to 9 rows answers an f64 dot within bf16
//! rounding: one row through the one-row vector forms, 2 to 8 through the
//! forms that read each weight row once for every row (a few-row decode
//! reads its weights once, as one row does), 9 through the row tile. Both
//! widths: up to 1024 columns a threadgroup a column, past it a simdgroup a
//! column; an odd contraction takes the scalar walk. No row past the rows is
//! written.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::linear::gemm;
use poem_ir::Dtype;
use crate::common::{bf16, noise};

fn value(at: u64, scale: f32) -> f32 {
    let v = (f32::from(noise(at)) - 127.5) * scale;
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

fn read_bf16(bytes: &[u8], at: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]])) << 16)
}

#[test]
fn a_few_row_dense_matmul_reads_each_weight_once() {
    let device = Context::bind().expect("a Metal 4 device");
    let tuning = engine_metal::chip::tuning("", false);
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a buffer");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    // (contraction, columns): narrow (a threadgroup a column), wide (a
    // simdgroup a column), and an odd contraction.
    for &(k, n) in &[(1024u32, 512u32), (1024, 3072), (1023, 640), (2048, 1024)] {
        let weights: Vec<u8> = (0..u64::from(n) * u64::from(k))
            .flat_map(|at| bf16(value(at ^ 0x77, 0.0009)))
            .collect();
        let w_b = upload(&weights);
        let w = Tensor::new(bind(&w_b), n, k, Dtype::Bf16);
        for m in 1..=9u32 {
            let inputs: Vec<u8> = (0..u64::from(m) * u64::from(k))
                .flat_map(|at| bf16(value(at ^ 0x5a5a_0000 ^ u64::from(m), 0.031)))
                .collect();
            let act_b = upload(&inputs);
            let act = Tensor::new(bind(&act_b), m, k, Dtype::Bf16);
            let bytes = u64::from(m) * u64::from(n) * 2;
            let out = upload(&vec![0xa5; bytes as usize + 4096]);
            let y = Tensor::new(bind(&out), m, n, Dtype::Bf16);
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            gemm::matmul(&sink, &tuning, act, w, y).expect("matmul");
            frame.commit().expect("commit");
            let got = handles.read(y.buf, bytes + 4096).expect("read");
            assert!(
                got[bytes as usize..].iter().all(|&b| b == 0xa5),
                "a write past the {m} rows at K={k} N={n}"
            );
            for row in 0..m {
                for col in (0..n).step_by(7) {
                    let (mut want, mut magnitude) = (0.0f64, 0.0f64);
                    for at in 0..k {
                        let a = f64::from(read_bf16(&inputs, (row * k + at) as usize));
                        let b = f64::from(read_bf16(&weights, (col * k + at) as usize));
                        want += a * b;
                        magnitude += (a * b).abs();
                    }
                    let got = f64::from(read_bf16(&got, (row * n + col) as usize));
                    let allowance = want.abs() / 256.0
                        + f64::from(k) * f64::from(f32::EPSILON) * magnitude
                        + 1e-6;
                    assert!(
                        (got - want).abs() <= allowance,
                        "M={m} K={k} N={n} row={row} col={col}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
