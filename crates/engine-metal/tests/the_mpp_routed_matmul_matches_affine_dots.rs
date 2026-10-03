#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::moe::{self, RoutedScratch};
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
fn every_routed_row_answers_its_experts_exact_affine_reference() {
    let Ok(device) = Context::bind() else {
        return;
    };
    // MPP ships with Metal 4, which is what the elastic pool asks of the device too.
    if !device.supports_elastic() {
        return;
    }
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let tuning = kernels_metal::DeviceTuning {
        qmm_mpp: true,
        moe_batch_min_pairs: 1,
        ..Default::default()
    };
    let mut checked = 0;
    // (experts, tokens, fan-out, K, N): row tiles of 16, 32 and 8.
    let cases = [
        (6u32, 40u32, 2u32, 128u32, 128u32),
        (4, 200, 2, 256, 256),
        (16, 12, 1, 128, 64),
    ];
    for exact in [true, false] {
        for &(experts, tokens, top_k, k, n) in &cases {
            let (groups, pairs) = (k / 64, tokens * top_k);
            let columns = experts * n;
            let codes: Vec<u8> = (0..u64::from(k) * u64::from(columns) / 2)
                .map(noise)
                .collect();
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
            let route = |token: u32, slot: u32| (token * 3 + slot * 5 + token / 7) % experts;
            let upload = |bytes: &[u8]| {
                let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
                buffer.write(0, bytes).expect("upload");
                buffer
            };
            let scales: Vec<u8> = (0..columns * groups)
                .flat_map(|at| bf16(scale(at)))
                .collect();
            let biases: Vec<u8> = (0..columns * groups)
                .flat_map(|at| bf16(bias(at)))
                .collect();
            let inputs: Vec<u8> = (0..tokens)
                .flat_map(|row| (0..k).flat_map(move |at| bf16(activation(row, at))))
                .collect();
            let routes: Vec<u8> = (0..tokens)
                .flat_map(|token| (0..top_k).flat_map(move |slot| route(token, slot).to_le_bytes()))
                .collect();
            let sorted = moe::sorted_rows(pairs, experts, &tuning);
            let output_bytes = u64::from(pairs) * u64::from(n) * 2;
            let planes = [
                upload(&codes),
                upload(&scales),
                upload(&biases),
                upload(&inputs),
                upload(&routes),
                upload(&vec![0xa5; output_bytes as usize + 64]),
                Buffer::zeroed(&device, u64::from(sorted) * 4).expect("perm"),
                Buffer::zeroed(&device, u64::from(sorted) * 4).expect("row experts"),
                Buffer::zeroed(&device, u64::from(sorted) * 4).expect("tile experts"),
                Buffer::zeroed(&device, u64::from(pairs) * 4).expect("inverse"),
                Buffer::zeroed(&device, u64::from(sorted) * u64::from(k) * 2).expect("sorted x"),
                Buffer::zeroed(&device, u64::from(sorted) * u64::from(n) * 2).expect("sorted y"),
                Buffer::zeroed(&device, u64::from(sorted) * u64::from(groups) * 4).expect("sums"),
            ];
            let at = |plane: usize, rows: u32, width: u32, dtype: Dtype| {
                let buffer = &planes[plane];
                let handle = handles.bind(buffer, 0, buffer.bytes()).expect("bind");
                Tensor::new(handle, rows, width, dtype)
            };
            let bank = Bank {
                mpp_codes: None,
                codes: at(0, columns, k, Dtype::U4g64),
                scales: at(1, columns, groups, Dtype::Bf16),
                biases: Some(at(2, columns, groups, Dtype::Bf16)),
                group: 64,
                bits: 4,
            };
            let y = at(5, pairs, n, Dtype::Bf16);
            let frame = device.frame().expect("frame");
            let batched = moe::matmul_select_batched(
                &Sink::new(&device, &frame, &pipelines, &handles),
                "the routed matmul",
                at(3, tokens, k, Dtype::Bf16),
                bank,
                None,
                at(4, tokens, top_k, Dtype::I32),
                experts,
                RoutedScratch {
                    perm: at(6, 1, sorted, Dtype::I32),
                    row_expert: at(7, 1, sorted, Dtype::I32),
                    tile_expert: at(8, 1, sorted, Dtype::I32),
                    inv: at(9, 1, pairs, Dtype::I32),
                    x: at(10, sorted, k, Dtype::Bf16),
                    y: at(11, sorted, n, Dtype::Bf16),
                    sums: at(12, sorted, groups, Dtype::F32),
                },
                y,
                &tuning,
            )
            .expect("the routed matmul fires");
            assert!(batched, "these routes are wide enough to batch");
            frame.commit().expect("commit");
            let got = handles.read(y.buf, output_bytes + 64).expect("read");
            assert!(
                got[output_bytes as usize..].iter().all(|&b| b == 0xa5),
                "write beyond the routed rows at K={k}, N={n}"
            );
            for token in 0..tokens {
                for slot in 0..top_k {
                    let expert = route(token, slot);
                    for col in 0..n {
                        let bank_col = expert * n + col;
                        let mut want = 0.0f64;
                        let mut absolute_sum = 0.0f64;
                        for at in 0..k {
                            let code_at = bank_col * k + at;
                            let q = (codes[(code_at / 2) as usize] >> ((code_at % 2) * 4)) & 15;
                            let factor = bank_col * groups + at / 64;
                            let product = f64::from(activation(token, at))
                                * (f64::from(q) * f64::from(scale(factor))
                                    + f64::from(bias(factor)));
                            want += product;
                            absolute_sum += product.abs();
                        }
                        let bits = (want as f32).to_bits();
                        let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                        let offset = (((token * top_k + slot) * n + col) * 2) as usize;
                        let actual = u16::from_le_bytes([got[offset], got[offset + 1]]);
                        let actual_f32 = f32::from_bits(u32::from(actual) << 16);
                        if exact {
                            assert_eq!(
                                actual, rounded,
                                "experts={experts} token={token} slot={slot} col={col}: {actual_f32} vs {want}"
                            );
                        } else {
                            let allowance = 0.00002
                                + f64::from(groups) * f64::from(f32::EPSILON) * absolute_sum;
                            assert!(
                                (f64::from(actual_f32) - want).abs()
                                    <= 0.004 * want.abs() + allowance,
                                "random experts={experts} token={token} slot={slot} col={col}: {actual_f32} vs {want}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!("checked {checked} routed outputs against exact and random affine dots");
}
