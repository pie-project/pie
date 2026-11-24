#![cfg(target_vendor = "apple")]

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

/// A dense bf16 matmul on the tensor ops (Apple10) answers the simdgroup
/// kernels' product within bf16 rounding of an f64 dot, for the shapes a
/// Qwen3 layer and head take and rows that end inside a tile, and writes no
/// row past its rows. A GPU without tensor ops takes the simdgroup kernels
/// either way, and the two answer bit for bit.
#[test]
fn a_dense_matmul_on_the_tensor_ops_answers_the_simdgroup_product() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let simdgroup = engine_metal::chip::tuning("", false);
    let tensor_ops = simdgroup.clone().with_family(device.family());
    let on_tensor_ops = tensor_ops.tensor_ops();
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a buffer");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    // (contraction, columns): a layer's packed qkv, o, gate-up and down, a
    // 64-column-only width, and the head.
    let shapes = [
        (1024u32, 4096u32),
        (2048, 1024),
        (1024, 6144),
        (3072, 1024),
        (512, 192),
        (1024, 151_936),
    ];
    let (mut worst_abs, mut worst_rel, mut worst_vs) = (0.0f64, 0.0f64, 0.0f64);
    let mut worst_simdgroup = 0.0f64;
    for &(k, n) in &shapes {
        let weights: Vec<u8> = (0..u64::from(n) * u64::from(k))
            .flat_map(|at| bf16(value(at ^ 0x77, 0.0009)))
            .collect();
        let w_b = upload(&weights);
        let w = Tensor::new(bind(&w_b), n, k, Dtype::Bf16);
        let rows_cases: &[u32] = if n > 100_000 {
            &[8, 64, 100]
        } else {
            &[4, 8, 9, 17, 20, 33, 64, 65, 100, 512]
        };
        for &m in rows_cases {
            let inputs: Vec<u8> = (0..u64::from(m) * u64::from(k))
                .flat_map(|at| bf16(value(at ^ 0x5a5a_0000, 0.031)))
                .collect();
            let act_b = upload(&inputs);
            let act = Tensor::new(bind(&act_b), m, k, Dtype::Bf16);
            let bytes = u64::from(m) * u64::from(n) * 2;
            let run = |tuning: &kernels_metal::Tuning| {
                let out = upload(&vec![0xa5; bytes as usize + 4096]);
                let y = Tensor::new(bind(&out), m, n, Dtype::Bf16);
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                gemm::matmul(&sink, tuning, act, w, y).expect("matmul");
                frame.commit().expect("commit");
                let got = handles.read(y.buf, bytes + 4096).expect("read");
                assert!(
                    got[bytes as usize..].iter().all(|&b| b == 0xa5),
                    "a write past the {m} rows at K={k} N={n}"
                );
                got
            };
            let base = run(&simdgroup);
            let ours = run(&tensor_ops);
            if !on_tensor_ops {
                assert_eq!(
                    base, ours,
                    "M={m} K={k} N={n}: no tensor ops, the same kernels"
                );
                continue;
            }
            let step = if n > 4096 { (n / 509) as usize } else { 1 };
            for row in 0..m {
                for col in (0..n).step_by(step) {
                    let (mut want, mut magnitude) = (0.0f64, 0.0f64);
                    for at in 0..k {
                        let a = f64::from(read_bf16(&inputs, (row * k + at) as usize));
                        let b = f64::from(read_bf16(&weights, (col * k + at) as usize));
                        want += a * b;
                        magnitude += (a * b).abs();
                    }
                    let at = (row * n + col) as usize;
                    let got = f64::from(read_bf16(&ours, at));
                    let simd = f64::from(read_bf16(&base, at));
                    let error = (got - want).abs();
                    // bf16's rounding of the sum, plus f32 accumulation over K.
                    let allowance = want.abs() / 256.0
                        + f64::from(k) * f64::from(f32::EPSILON) * magnitude
                        + 1e-6;
                    assert!(
                        error <= allowance,
                        "M={m} K={k} N={n} row={row} col={col}: {got} vs {want} (simdgroup {simd})"
                    );
                    worst_abs = worst_abs.max(error);
                    worst_rel = worst_rel.max(error / magnitude.max(1e-12));
                    worst_vs = worst_vs.max((got - simd).abs());
                    worst_simdgroup = worst_simdgroup.max((simd - want).abs());
                }
            }
        }
    }
    // Activations that end on a page, the buffer no longer than its rows:
    // the last row tile reads past `rows` with nothing mapped behind them.
    let (k, n) = (8192u32, 1024u32);
    let weights: Vec<u8> = (0..u64::from(n) * u64::from(k))
        .flat_map(|at| bf16(value(at ^ 0x99, 0.0009)))
        .collect();
    let w_b = upload(&weights);
    let w = Tensor::new(bind(&w_b), n, k, Dtype::Bf16);
    for m in [9u32, 100] {
        let bytes = u64::from(m) * u64::from(k) * 2;
        assert_eq!(
            bytes % engine_metal::mapping::page() as u64,
            0,
            "the activations end on a page"
        );
        let inputs: Vec<u8> = (0..u64::from(m) * u64::from(k))
            .flat_map(|at| bf16(value(at ^ 0x3c3c_0000, 0.031)))
            .collect();
        let act_b = upload(&inputs);
        assert_eq!(act_b.bytes(), bytes, "the buffer is exactly its rows");
        let out = upload(&vec![0xa5; (m * n * 2) as usize]);
        let y = Tensor::new(bind(&out), m, n, Dtype::Bf16);
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        gemm::matmul(
            &sink,
            &tensor_ops,
            Tensor::new(bind(&act_b), m, k, Dtype::Bf16),
            w,
            y,
        )
        .expect("matmul");
        frame
            .commit()
            .expect("an exact-size, page-end activation reads without a fault");
        let got = handles.read(y.buf, u64::from(m * n * 2)).expect("read");
        for row in [0, m - 1] {
            for col in (0..n).step_by(61) {
                let (mut want, mut magnitude) = (0.0f64, 0.0f64);
                for at in 0..k {
                    let a = f64::from(read_bf16(&inputs, (row * k + at) as usize));
                    let b = f64::from(read_bf16(&weights, (col * k + at) as usize));
                    want += a * b;
                    magnitude += (a * b).abs();
                }
                let actual = f64::from(read_bf16(&got, (row * n + col) as usize));
                let allowance =
                    want.abs() / 256.0 + f64::from(k) * f64::from(f32::EPSILON) * magnitude + 1e-6;
                assert!(
                    (actual - want).abs() <= allowance,
                    "page-end M={m} row={row} col={col}: {actual} vs {want}"
                );
            }
        }
    }
}
