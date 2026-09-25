#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

#[test]
fn every_prefill_row_answers_an_exact_affine_reference() {
    let Ok(device) = Context::bind() else {
        return;
    };
    assert!(kernels_metal::tuning::override_with(
        kernels_metal::tuning::Overrides {
            qmm_mpp: Some(true),
            ..Default::default()
        }
    ));
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut checked = 0;
    let cases = [
        (64u32, 128u32),
        (192, 256),
        (512, 384),
        (128, 160),
        (512, 1024),
        (5120, 34816),
        (17408, 5120),
        (5120, 12288),
        (5120, 16384),
        (6144, 5120),
    ];
    for exact in [true, false] {
        for &(k, n) in &cases {
            let groups = k / 64;
            let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2).map(noise).collect();
            let stored = |value: f32| f32::from_bits(value.to_bits() & 0xffff0000);
            let scale = |at: u32| {
                if exact {
                    ((at % 7) + 1) as f32 / 128.0
                } else {
                    stored(0.0031 + f32::from(noise(u64::from(at))) * 0.000173)
                }
            };
            let bias = |at: u32| {
                if exact {
                    ((at % 9) as i32 - 4) as f32 / 64.0
                } else {
                    stored((f32::from(noise(u64::from(at) ^ 0x5678)) - 127.0) * 0.00157)
                }
            };
            let activation = |row: u32, at: u32| {
                if exact {
                    (((row * 17 + at * 3) % 29) as i32 - 14) as f32 / 16.0
                } else {
                    stored((f32::from(noise(u64::from(row * k + at) ^ 0xabcd)) - 127.0) * 0.0197)
                }
            };
            let scales: Vec<u8> = (0..n * groups).flat_map(|at| bf16(scale(at))).collect();
            let biases: Vec<u8> = (0..n * groups).flat_map(|at| bf16(bias(at))).collect();
            let upload = |bytes: &[u8]| {
                let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
                buffer.write(0, bytes).expect("upload");
                buffer
            };
            let codes_b = upload(&codes);
            let scales_b = upload(&scales);
            let biases_b = upload(&biases);
            let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
            let packed_buffer = if n % 256 == 0 {
                let mut packed = vec![0u8; codes.len()];
                for col in 0..n {
                    for g in 0..groups {
                        let from = (col * k / 2 + g * 32) as usize;
                        let into = ((col / 256 * groups + g) * 256 * 32 + col % 256 * 32) as usize;
                        packed[into..into + 32].copy_from_slice(&codes[from..from + 32]);
                    }
                }
                for src in [&scales, &biases] {
                    let mut dst = vec![0u8; src.len()];
                    for col in 0..n {
                        for g in 0..groups {
                            let from = ((col * groups + g) * 2) as usize;
                            let into = (((col / 256 * groups + g) * 256 + col % 256) * 2) as usize;
                            dst[into..into + 2].copy_from_slice(&src[from..from + 2]);
                        }
                    }
                    packed.extend_from_slice(&dst);
                }
                let buffer = Buffer::zeroed(&device, packed.len() as u64).expect("GPU packed bank");
                let frame = device.frame().expect("pack frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                let source = Tensor::new(bind(&codes_b), n, k, Dtype::U4g64);
                let scale_t = Tensor::new(bind(&scales_b), n, groups, Dtype::Bf16);
                let bias_t = Tensor::new(bind(&biases_b), n, groups, Dtype::Bf16);
                let dest = Tensor::new(bind(&buffer), n, k, Dtype::U4g64);
                sink.fire(
                    Fire::at("linear/quant_qmm_mpp.metal", "qmm_mpp_pack_bank")
                        .apply(Grid::of([n * (k / 8), 1, 1], [256, 1, 1])),
                    &[
                        source.arg(),
                        scale_t.arg(),
                        bias_t.arg(),
                        dest.arg_mut(),
                        k.arg(),
                        n.arg(),
                    ],
                )
                .expect("pack");
                frame.commit().expect("pack complete");
                assert_eq!(
                    handles
                        .read(dest.buf, packed.len() as u64)
                        .expect("packed readback"),
                    packed,
                    "GPU layout preserves every code/scale/bias bit"
                );
                Some(buffer)
            } else {
                None
            };
            let bank = Bank {
                mpp_codes: packed_buffer
                    .as_ref()
                    .map(|b| Tensor::new(bind(b), n, k, Dtype::U4g64)),
                codes: Tensor::new(bind(&codes_b), n, k, Dtype::U4g64),
                scales: Tensor::new(bind(&scales_b), n, groups, Dtype::Bf16),
                biases: Some(Tensor::new(bind(&biases_b), n, groups, Dtype::Bf16)),
                group: 64,
                bits: 4,
            };
            for m in [8u32, 17, 32, 65, 2048] {
                if !exact && !(n % 128 == 0 && (m > 16 || n >= 1024)) {
                    continue;
                }
                if k > 512 && ![8, 2048].contains(&m) {
                    continue;
                }
                if k <= 512 && m == 2048 {
                    continue;
                }
                let cap = m.div_ceil(64) * 64;
                let inputs: Vec<u8> = (0..cap)
                    .flat_map(|row| (0..k).flat_map(move |at| bf16(activation(row, at))))
                    .collect();
                let act_b = upload(&inputs);
                let output_bytes = u64::from(cap) * u64::from(n) * 2;
                let out_b = upload(&vec![0xa5; output_bytes as usize + 64]);
                let staging =
                    Buffer::zeroed(&device, u64::from(cap) * u64::from(k) * 2).expect("staging");
                let (ha, ho, hs) = (bind(&act_b), bind(&out_b), bind(&staging));
                let precast = |rows, width| {
                    (rows <= cap && width == k).then(|| Tensor::new(hs, rows, width, Dtype::F16))
                };
                let partial_bytes = u64::from(cap) * 8 * u64::from(n) * 4;
                let partial_buffer = upload(&vec![0xa5; partial_bytes as usize + 64]);
                let ph = bind(&partial_buffer);
                let partials = |rows, width| {
                    (rows <= cap * 8 && width == n)
                        .then(|| Tensor::new(ph, rows, width, Dtype::F32))
                };
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                quant::matmul(
                    &sink,
                    Tensor::new(ha, m, k, Dtype::Bf16),
                    bank,
                    Tensor::new(ho, m, n, Dtype::Bf16),
                    quant::Scratch {
                        precast: &precast,
                        partials: &partials,
                    },
                    cap,
                )
                .expect("prefill");
                frame.commit().expect("commit");
                let got = handles.read(ho, output_bytes + 64).expect("read");
                let pp = handles.read(ph, partial_bytes + 64).expect("partial guard");
                assert!(
                    pp[partial_bytes as usize..].iter().all(|&b| b == 0xa5),
                    "partial output overflow"
                );
                assert!(
                    got[output_bytes as usize..].iter().all(|&b| b == 0xa5),
                    "write beyond output capacity at M={m}, K={k}, N={n}"
                );
                for row in 0..m {
                    for col in (0..n).step_by(if n > 1024 { (n / 32) as usize } else { 1 }) {
                        let mut want = 0.0f64;
                        let mut absolute_sum = 0.0f64;
                        for at in 0..k {
                            let code_at = col * k + at;
                            let q = (codes[(code_at / 2) as usize] >> ((code_at % 2) * 4)) & 15;
                            let factor = col * groups + at / 64;
                            let product = f64::from(activation(row, at))
                                * (f64::from(q) * f64::from(scale(factor))
                                    + f64::from(bias(factor)));
                            want += product;
                            absolute_sum += product.abs();
                        }
                        let bits = (want as f32).to_bits();
                        let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                        let offset = ((row * n + col) * 2) as usize;
                        let actual = u16::from_le_bytes([got[offset], got[offset + 1]]);
                        let actual_f32 = f32::from_bits(u32::from(actual) << 16);
                        if exact {
                            assert_eq!(
                                actual, rounded,
                                "M={m} K={k} N={n} row={row} col={col}: {actual_f32} vs {want}"
                            );
                        } else {
                            let allowance = 0.00002
                                + f64::from(groups) * f64::from(f32::EPSILON) * absolute_sum;
                            assert!(
                                (f64::from(actual_f32) - want).abs()
                                    <= 0.004 * want.abs() + allowance,
                                "random M={m} K={k} N={n} row={row} col={col}: {actual_f32} vs {want}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "checked {checked} prefill outputs against exact and random affine dots, including row tails and a fallback column width"
    );
}
