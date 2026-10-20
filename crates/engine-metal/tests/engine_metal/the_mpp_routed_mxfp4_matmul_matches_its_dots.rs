#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::moe::{self, RoutedScratch};
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

const LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

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
fn every_routed_row_answers_its_experts_exact_mxfp4_reference() {
    let Ok(device) = Context::bind() else {
        return;
    };
    if !device.supports_elastic() {
        return;
    }
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let tuning = engine_metal::chip::tuning(
        r#"
[kernel]
moe.path = "mpp"
moe.tile_rows = [{ when = "per_expert >= 64", value = 64 }, { value = 32 }]
"#,
        false,
    );
    let mut checked = 0;
    // (experts, tokens, fan-out, K, N, codes as one row an expert): row tiles of 64 and 32.
    let cases = [
        (4u32, 160u32, 2u32, 128u32, 128u32, false),
        (6, 120, 2, 256, 192, true),
    ];
    for exact in [true, false] {
        for biased in [false, true] {
            for &(experts, tokens, top_k, k, n, flat) in &cases {
                let (blocks, pairs) = (k / 32, tokens * top_k);
                let columns = experts * n;
                let codes: Vec<u8> = (0..u64::from(k) * u64::from(columns) / 2)
                    .map(noise)
                    .collect();
                let stored = |value: f32| f32::from_bits(value.to_bits() & 0xffff0000);
                let exponent = |at: u32| {
                    let spread = if exact { 3 } else { 9 };
                    127 - 3 + (u32::from(noise(u64::from(at) ^ 0x77)) % spread) as u8
                };
                let bias = |at: u32| {
                    if exact {
                        ((at % 9) as i32 - 4) as f32 / 64.0
                    } else {
                        stored((f32::from(noise(u64::from(at) ^ 0x5678)) - 127.0) * 0.0157)
                    }
                };
                let activation = |row: u32, at: u32| {
                    if exact {
                        (((row * 17 + at * 3) % 29) as i32 - 14) as f32 / 16.0
                    } else {
                        stored(
                            (f32::from(noise(u64::from(row * k + at) ^ 0xabcd)) - 127.0) * 0.0197,
                        )
                    }
                };
                let route = |token: u32, slot: u32| (token * 3 + slot * 5 + token / 7) % experts;
                let upload = |bytes: &[u8]| {
                    let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
                    buffer.write(0, bytes).expect("upload");
                    buffer
                };
                let exponents: Vec<u8> = (0..columns * blocks).map(exponent).collect();
                let biases: Vec<u8> = (0..columns).flat_map(|at| bf16(bias(at))).collect();
                let inputs: Vec<u8> = (0..tokens)
                    .flat_map(|row| (0..k).flat_map(move |at| bf16(activation(row, at))))
                    .collect();
                let routes: Vec<u8> = (0..tokens)
                    .flat_map(|token| {
                        (0..top_k).flat_map(move |slot| route(token, slot).to_le_bytes())
                    })
                    .collect();
                let sorted = moe::sorted_rows(pairs, experts, &tuning);
                let output_bytes = u64::from(pairs) * u64::from(n) * 2;
                let planes = [
                    upload(&codes),
                    upload(&exponents),
                    upload(&biases),
                    upload(&inputs),
                    upload(&routes),
                    upload(&vec![0xa5; output_bytes as usize + 64]),
                    Buffer::zeroed(&device, u64::from(sorted) * 4).expect("perm"),
                    Buffer::zeroed(&device, u64::from(sorted) * 4).expect("row experts"),
                    Buffer::zeroed(&device, u64::from(sorted) * 4).expect("tile experts"),
                    Buffer::zeroed(&device, u64::from(pairs) * 4).expect("inverse"),
                    Buffer::zeroed(&device, u64::from(sorted) * u64::from(k) * 2)
                        .expect("sorted x"),
                    Buffer::zeroed(&device, u64::from(sorted) * u64::from(n) * 2)
                        .expect("sorted y"),
                    Buffer::zeroed(&device, u64::from(sorted) * u64::from(k / 64) * 4)
                        .expect("sums"),
                ];
                let at = |plane: usize, rows: u32, width: u32, dtype: Dtype| {
                    let buffer = &planes[plane];
                    let handle = handles.bind(buffer, 0, buffer.bytes()).expect("bind");
                    Tensor::new(handle, rows, width, dtype)
                };
                // A bank states its codes as K-wide expert columns or as one row an expert.
                let bank = Bank {
                    mpp_codes: None,
                    codes: if flat {
                        at(0, experts, n * k / 2, Dtype::Mxfp4)
                    } else {
                        at(0, columns, k, Dtype::U4g64)
                    },
                    scales: at(1, columns, blocks, Dtype::E8m0),
                    biases: None,
                    group: 32,
                    bits: 4,
                    ld: if flat { n * k / 2 } else { k },
                };
                let y = at(5, pairs, n, Dtype::Bf16);
                let frame = device.frame().expect("frame");
                let batched = moe::matmul_select_batched(
                    &Sink::new(&device, &frame, &pipelines, &handles),
                    "the routed matmul",
                    at(3, tokens, k, Dtype::Bf16),
                    bank,
                    biased.then(|| at(2, experts, n, Dtype::Bf16)),
                    at(4, tokens, top_k, Dtype::I32),
                    experts,
                    RoutedScratch {
                        perm: at(6, 1, sorted, Dtype::I32),
                        row_expert: at(7, 1, sorted, Dtype::I32),
                        tile_expert: at(8, 1, sorted, Dtype::I32),
                        inv: at(9, 1, pairs, Dtype::I32),
                        x: at(10, sorted, k, Dtype::Bf16),
                        y: at(11, sorted, n, Dtype::Bf16),
                        sums: at(12, sorted, k / 64, Dtype::F32),
                    },
                    y,
                    &tuning,
                    moe::Sorted::default(),
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
                                let e = exponents[(bank_col * blocks + at / 32) as usize];
                                let weight =
                                    f64::from(LUT[q as usize]) * 2f64.powi(i32::from(e) - 127);
                                let product = f64::from(activation(token, at)) * weight;
                                want += product;
                                absolute_sum += product.abs();
                            }
                            if biased {
                                want += f64::from(bias(bank_col));
                            }
                            let bits = (want as f32).to_bits();
                            let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                            let offset = (((token * top_k + slot) * n + col) * 2) as usize;
                            let actual = u16::from_le_bytes([got[offset], got[offset + 1]]);
                            let actual_f32 = f32::from_bits(u32::from(actual) << 16);
                            if exact {
                                assert_eq!(
                                    actual, rounded,
                                    "experts={experts} bias={biased} token={token} slot={slot} col={col}: {actual_f32} vs {want}"
                                );
                            } else {
                                let allowance = 0.00002
                                    + f64::from(blocks) * f64::from(f32::EPSILON) * absolute_sum;
                                assert!(
                                    (f64::from(actual_f32) - want).abs()
                                        <= 0.004 * want.abs() + allowance,
                                    "random experts={experts} bias={biased} token={token} slot={slot} col={col}: {actual_f32} vs {want}"
                                );
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("checked {checked} routed outputs against exact and random mxfp4 dots");
}
