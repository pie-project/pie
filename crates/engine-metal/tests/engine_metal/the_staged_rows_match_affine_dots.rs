#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

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
fn every_staged_row_answers_an_affine_reference() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let tuning = engine_metal::chip::tuning("[kernel]\nqmm = [\"staged\", \"qmv\"]", false);
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut checked = 0;
    let cases = [
        (128u32, 64u32),
        (512, 1024),
        (5120, 1280),
        (17408, 5120),
        (5120, 34816),
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
            let bank = Bank {
                mpp_codes: None,
                codes: Tensor::new(bind(&codes_b), n, k, Dtype::U4g64),
                scales: Tensor::new(bind(&scales_b), n, groups, Dtype::Bf16),
                biases: Some(Tensor::new(bind(&biases_b), n, groups, Dtype::Bf16)),
                group: 64,
                bits: 4,
                ld: k,
            };
            for m in [5u32, 6, 8] {
                let cap = 8;
                let inputs: Vec<u8> = (0..cap)
                    .flat_map(|row| (0..k).flat_map(move |at| bf16(activation(row, at))))
                    .collect();
                let act_b = upload(&inputs);
                let output_bytes = u64::from(m) * u64::from(n) * 2;
                let out_b = upload(&vec![0xa5; output_bytes as usize + 64]);
                let (ha, ho) = (bind(&act_b), bind(&out_b));
                let precast = |_rows, _width| None;
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
                    &tuning,
                    Tensor::new(ha, m, k, Dtype::Bf16),
                    bank,
                    Tensor::new(ho, m, n, Dtype::Bf16),
                    quant::Scratch {
                        precast: &precast,
                        partials: &partials,
                        table: None,
                    },
                    cap,
                )
                .expect("staged");
                frame.commit().expect("commit");
                let got = handles.read(ho, output_bytes + 64).expect("read");
                assert!(
                    got[output_bytes as usize..].iter().all(|&b| b == 0xa5),
                    "write beyond output at M={m}, K={k}, N={n}"
                );
                for row in 0..m {
                    for col in (0..n).step_by(if n > 1024 { (n / 64) as usize } else { 1 }) {
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
                            let allowance = 0.00002 + 0.001 * absolute_sum / f64::from(k).sqrt();
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
    eprintln!("checked {checked} staged outputs against exact and random affine dots");
}

#[test]
fn the_coop_swiglu_is_the_matmul_then_the_swiglu() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let tuning = engine_metal::chip::tuning("[kernel]\nqmm = [\"staged\", \"qmv\"]", false);
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    for &(k, intermediate) in &[(1024u32, 6144u32), (5120, 6400)] {
        let (n, groups) = (2 * intermediate, k / 64);
        let codes = upload(
            &(0..u64::from(k) * u64::from(n) / 2)
                .map(noise)
                .collect::<Vec<_>>(),
        );
        let factor = |at: u32, salt: u64, base: f32, step: f32| {
            bf16(base + (f32::from(noise(u64::from(at) ^ salt)) - 127.0) * step)
        };
        let scales = upload(
            &(0..n * groups)
                .flat_map(|at| factor(at, 0x11, 0.004, 0.00002))
                .collect::<Vec<_>>(),
        );
        let biases = upload(
            &(0..n * groups)
                .flat_map(|at| factor(at, 0x22, 0.0, 0.0002))
                .collect::<Vec<_>>(),
        );
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(bind(&codes), n, k, Dtype::U4g64),
            scales: Tensor::new(bind(&scales), n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&biases), n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        for m in [5u32, 6, 8] {
            let x = upload(
                &(0..m * k)
                    .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0x33)) - 127.0) / 128.0))
                    .collect::<Vec<_>>(),
            );
            let packed = upload(&vec![0; (m * n * 2) as usize]);
            let (two, one) = (
                upload(&vec![0; (m * intermediate * 2) as usize]),
                upload(&vec![0xa5; (m * intermediate * 2) as usize + 64]),
            );
            let act = Tensor::new(bind(&x), m, k, Dtype::Bf16);
            let packed = Tensor::new(bind(&packed), m, n, Dtype::Bf16);
            let (two, one) = (
                Tensor::new(bind(&two), m, intermediate, Dtype::Bf16),
                Tensor::new(bind(&one), m, intermediate, Dtype::Bf16),
            );
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            quant::matmul(
                &sink,
                &tuning,
                act,
                bank,
                packed,
                quant::Scratch {
                    precast: &|_, _| None,
                    partials: &|_, _| None,
                    table: None,
                },
                8,
            )
            .expect("gate and up");
            kernels_metal::linear::mlp::swiglu(&sink, packed, intermediate, two).expect("swiglu");
            assert!(
                quant::coop_swiglu(&sink, &tuning, act, bank, intermediate, one, 8).expect("fused"),
                "the fused tile takes K={k} I={intermediate}"
            );
            frame.commit().expect("commit");
            let bytes = u64::from(m * intermediate) * 2;
            let (want, got) = (
                handles.read(two.buf, bytes).expect("read"),
                handles.read(one.buf, bytes + 64).expect("read"),
            );
            assert!(
                got[bytes as usize..].iter().all(|&b| b == 0xa5),
                "a write past h"
            );
            assert!(
                want == got[..bytes as usize],
                "M={m} K={k} I={intermediate}: the fused tile differs from the two launches"
            );
            assert!(want.iter().any(|&b| b != 0), "the swiglu is not all zeros");
        }
    }
}

#[test]
fn the_coop_segments_are_their_matmuls() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len().max(16) as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let k = 2048u32;
    let groups = k / 64;
    // (columns, the segment's first row, its width): a split bank's halves (the second entry's
    // own bank unused), and a narrow one
    let banks = [(6144u32, 0u32, 4096u32), (6144, 4096, 2048), (96, 0, 96)];
    let mut keep = Vec::new();
    let mut planes = Vec::new();
    for (b, &(n, _, _)) in banks.iter().enumerate() {
        let salt = b as u64 * 0x1000;
        let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2)
            .map(|at| noise(at ^ salt))
            .collect();
        let scales: Vec<f32> = (0..n * groups)
            .map(|at| {
                let v = 0.004 + (f32::from(noise(u64::from(at) ^ salt ^ 0x11)) - 127.0) * 0.00002;
                f32::from_bits(v.to_bits() & 0xffff_0000)
            })
            .collect();
        let biases: Vec<f32> = (0..n * groups)
            .map(|at| {
                let v = (f32::from(noise(u64::from(at) ^ salt ^ 0x22)) - 127.0) * 0.0002;
                f32::from_bits(v.to_bits() & 0xffff_0000)
            })
            .collect();
        let cb = upload(&codes);
        let sb = upload(&scales.iter().flat_map(|&v| bf16(v)).collect::<Vec<_>>());
        let bb = upload(&biases.iter().flat_map(|&v| bf16(v)).collect::<Vec<_>>());
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(bind(&cb), n, k, Dtype::U4g64),
            scales: Tensor::new(bind(&sb), n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&bb), n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        keep.extend([cb, sb, bb]);
        planes.push((bank, codes, scales, biases));
    }
    for m in [5u32, 8] {
        let input: Vec<f32> = (0..m * k)
            .map(|at| {
                let v = (f32::from(noise(u64::from(at) ^ 0x33)) - 127.0) / 128.0;
                f32::from_bits(v.to_bits() & 0xffff_0000)
            })
            .collect();
        let xb = upload(&input.iter().flat_map(|&v| bf16(v)).collect::<Vec<_>>());
        let act = Tensor::new(bind(&xb), m, k, Dtype::Bf16);
        let outs: Vec<Buffer> = banks
            .iter()
            .map(|&(_, _, w)| upload(&vec![0xa5; (m * w * 2) as usize]))
            .collect();
        let segments: Vec<quant::Segment> = banks
            .iter()
            .zip(&planes)
            .zip(&outs)
            .enumerate()
            .map(|(i, ((&(_, first, w), plane), out))| quant::Segment {
                // the two halves of one bank
                w: if i == 1 { planes[0].0 } else { plane.0 },
                first,
                y: Tensor::new(bind(out), m, w, Dtype::Bf16),
                split: 1,
            })
            .collect();
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        quant::coop_segments(&sink, &segments, act, m as i32, k as i32).expect("segments");
        // the 6144-row bank alone, unsplit, for the halves to match
        let whole = upload(&vec![0; (m * 6144 * 2) as usize]);
        let whole_t = Tensor::new(bind(&whole), m, 6144, Dtype::Bf16);
        quant::coop_matmul(
            &sink,
            act,
            planes[0].0,
            planes[0].0.biases.expect("biases"),
            whole_t,
            (m as i32, 6144, k as i32, 6144, k as i32),
            1,
        )
        .expect("alone");
        frame.commit().expect("commit");
        let whole = handles
            .read(whole_t.buf, u64::from(m * 6144) * 2)
            .expect("read");
        for (s, &(_, first, w)) in segments.iter().zip(&banks).take(2) {
            let got = handles.read(s.y.buf, u64::from(m * w) * 2).expect("read");
            for row in 0..m as usize {
                let at = (row * 6144 + first as usize) * 2;
                assert!(
                    got[row * w as usize * 2..(row + 1) * w as usize * 2]
                        == whole[at..at + w as usize * 2],
                    "M={m}: the half from {first} differs from the unsplit tile's"
                );
            }
        }
        // the narrow segment against an fp64 dot of its dequantized weights
        let (_, codes, scales, biases) = &planes[2];
        let got = handles
            .read(segments[2].y.buf, u64::from(m * 96) * 2)
            .expect("read");
        for row in 0..m {
            for col in 0..96u32 {
                let (mut want, mut absolute) = (0.0f64, 0.0f64);
                for at in 0..k {
                    let code_at = col * k + at;
                    let q = (codes[(code_at / 2) as usize] >> ((code_at % 2) * 4)) & 15;
                    let f = (col * groups + at / 64) as usize;
                    let p = f64::from(input[(row * k + at) as usize])
                        * (f64::from(q) * f64::from(scales[f]) + f64::from(biases[f]));
                    want += p;
                    absolute += p.abs();
                }
                let o = ((row * 96 + col) * 2) as usize;
                let actual = f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([got[o], got[o + 1]])) << 16,
                ));
                let allowance = 0.00002 + 0.001 * absolute / f64::from(k).sqrt();
                assert!(
                    (actual - want).abs() <= 0.004 * want.abs() + allowance,
                    "M={m} narrow col={col} row={row}: {actual} vs {want}"
                );
            }
        }
    }
    drop(keep);
}
