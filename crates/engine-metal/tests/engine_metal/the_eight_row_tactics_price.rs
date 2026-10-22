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
#[ignore = "a microbenchmark"]
fn the_eight_row_tactics_price() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let tactics = [
        ("register", "[kernel]\nqmm = [\"register\", \"qmv\"]"),
        ("staged", "[kernel]\nqmm = [\"staged\", \"qmv\"]"),
        ("tiled96", "[kernel]\nqmm = [{ use = \"tiled\", crossover_tg = 96 }, \"qmv\"]"),
        ("qmv", "[kernel]\nqmm = [{ use = \"qmv\", rows_max = 8, packs = 1, crossover_tg = 96 }]"),
        ("qmv-p2", "[kernel]\nqmm = [{ use = \"qmv\", rows_max = 8, packs = 2, crossover_tg = 96 }]"),
        ("qmv-r4", "[kernel]\nqmm = [{ use = \"qmv\", rows_max = 4, packs = 1, crossover_tg = 96 }]"),
        ("mpp", "[kernel]\nqmm = [\"mpp\", \"register\", \"qmv\"]"),
    ];
    // (k, n): the 35B verify's GDN in-projection, out-projections, attention
    // in-projection, shared expert gate-up and down, router, head; the draft's MLP.
    let cases = [
        (2048u32, 12288u32),
        (4096, 2048),
        (2048, 9216),
        (2048, 1024),
        (512, 2048),
        (2048, 256),
        (2048, 248320),
        (2048, 12288 / 2),
        (6144, 2048),
    ];
    for m in [8u32, 1] {
        for &(k, n) in &cases {
            let groups = k / 64;
            let bank_bytes = u64::from(k) * u64::from(n) / 2;
            // enough distinct banks that a pass streams 256 MB, as a round does
            let copies = (256u64 << 20).div_ceil(bank_bytes).clamp(1, 64) as usize;
            let mut planes = Vec::new();
            let mut banks = Vec::new();
            for c in 0..copies {
                let seed = c as u64 * 0x1_0000_0000;
                let codes: Vec<u8> = (0..bank_bytes).map(|at| noise(at ^ seed)).collect();
                let scales: Vec<u8> = (0..n * groups)
                    .flat_map(|at| bf16(0.0031 + f32::from(noise(u64::from(at) ^ seed ^ 0x11)) * 0.000173))
                    .collect();
                let biases: Vec<u8> = (0..n * groups)
                    .flat_map(|at| bf16(-0.2 + f32::from(noise(u64::from(at) ^ seed ^ 0x5678)) * 0.00157))
                    .collect();
                let p = [upload(&codes), upload(&scales), upload(&biases)];
                banks.push(Bank {
                    mpp_codes: None,
                    codes: Tensor::new(bind(&p[0]), n, k, Dtype::U4g64),
                    scales: Tensor::new(bind(&p[1]), n, groups, Dtype::Bf16),
                    biases: Some(Tensor::new(bind(&p[2]), n, groups, Dtype::Bf16)),
                    group: 64,
                    bits: 4,
                    ld: k,
                });
                planes.push(p);
            }
            let cap = 8;
            let inputs: Vec<u8> = (0..cap * k)
                .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0xabcd)) - 127.0) * 0.0197))
                .collect();
            let act_b = upload(&inputs);
            let out_b = upload(&vec![0u8; (cap * n * 2) as usize]);
            let (ha, ho) = (bind(&act_b), bind(&out_b));
            let table_rows = 16 + (4 * n / 64).div_ceil(2 * k);
            let table_b = upload(&vec![0u8; (table_rows * k * 2) as usize]);
            let th = bind(&table_b);
            let partial_b = upload(&vec![0u8; (cap * 8 * n * 4) as usize + 64]);
            let ph = bind(&partial_b);
            let precast = |rows: u32, width: u32| {
                (rows <= table_rows && width == k).then(|| Tensor::new(th, rows, width, Dtype::F16))
            };
            let partials = |rows: u32, width: u32| {
                (rows <= cap * 8 && width <= n).then(|| Tensor::new(ph, rows, width, Dtype::F32))
            };
            let mut line = format!("M={m} K={k:>5} N={n:>6} x{copies:>2}:");
            for (name, overrides) in &tactics {
                let tuning = engine_metal::chip::tuning(overrides, false);
                let mut times = Vec::new();
                let mut failed = false;
                for i in 0..12 {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    for bank in &banks {
                        if quant::matmul(
                            &sink,
                            &tuning,
                            Tensor::new(ha, m, k, Dtype::Bf16),
                            *bank,
                            Tensor::new(ho, m, n, Dtype::Bf16),
                            quant::Scratch {
                                precast: &precast,
                                partials: &partials,
                                table: None,
                            },
                            cap,
                        )
                        .is_err()
                        {
                            failed = true;
                            break;
                        }
                    }
                    let s = frame.commit_timed().expect("commit");
                    if failed {
                        break;
                    }
                    if i >= 2 {
                        times.push(s * 1e6 / copies as f64);
                    }
                }
                if failed {
                    line += &format!("  {name} -");
                    continue;
                }
                times.sort_by(f64::total_cmp);
                let us = times[times.len() / 2];
                let gbs = (bank_bytes as f64 * 1.0625) / (us * 1e3);
                line += &format!("  {name} {us:.1}us {gbs:.0}GB/s");
            }
            eprintln!("{line}");
        }
    }
}

/// An 8-row verify's routed matmuls of the 35B (256 experts, top 8): the vector
/// kernel a pair at a time against the sorted tiles, over routes that touch
/// `distinct` experts, each of 40 layers its own experts.
#[test]
#[ignore = "a microbenchmark"]
fn the_eight_row_routed_tactics_price() {
    use kernels_metal::linear::moe::{self, RoutedScratch};
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let (experts, tokens, top_k, layers) = (256u32, 8u32, 8u32, 40u32);
    let pairs = tokens * top_k;
    let tactics: [(&str, &str); 6] = [
        ("qmv", ""),
        ("tiled8", "[kernel]\nmoe.path = \"tiled\"\nmoe.tile_rows = [{ value = 8 }]"),
        ("tiled16", "[kernel]\nmoe.path = \"tiled\"\nmoe.tile_rows = [{ value = 16 }]"),
        ("mpp8", "[kernel]\nmoe.path = \"mpp\"\nmoe.tile_rows = [{ value = 8 }]"),
        ("mpp16", "[kernel]\nmoe.path = \"mpp\"\nmoe.tile_rows = [{ value = 16 }]"),
        ("tiled8nofp16", "[kernel]\nqmm_fp16 = false\nmoe.path = \"tiled\"\nmoe.tile_rows = [{ value = 8 }]"),
    ];
    // gate-up (K 2048 -> N 1024 a token) and down (K 512 a pair -> N 2048)
    for &(k, n, per_pair) in &[(2048u32, 1024u32, false), (512, 2048, true)] {
        let groups = k / 64;
        let columns = experts * n;
        let bank_bytes = u64::from(k) * u64::from(columns) / 2;
        let codes: Vec<u8> = (0..bank_bytes).map(noise).collect();
        let scales: Vec<u8> = (0..columns * groups)
            .flat_map(|at| bf16(0.0031 + f32::from(noise(u64::from(at) ^ 0x11)) * 0.000173))
            .collect();
        let biases: Vec<u8> = (0..columns * groups)
            .flat_map(|at| bf16(-0.2 + f32::from(noise(u64::from(at) ^ 0x5678)) * 0.00157))
            .collect();
        let p = [upload(&codes), upload(&scales), upload(&biases)];
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(bind(&p[0]), columns, k, Dtype::U4g64),
            scales: Tensor::new(bind(&p[1]), columns, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&p[2]), columns, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        let x_rows = if per_pair { pairs } else { tokens };
        let inputs: Vec<u8> = (0..x_rows * k)
            .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0xabcd)) - 127.0) * 0.0197))
            .collect();
        let x_b = upload(&inputs);
        let x = Tensor::new(bind(&x_b), x_rows, k, Dtype::Bf16);
        let y_b = upload(&vec![0u8; (pairs * n * 2) as usize]);
        let y = Tensor::new(bind(&y_b), pairs, n, Dtype::Bf16);
        let sorted = 1024u32;
        let scratch_b = [
            Buffer::zeroed(&device, u64::from(sorted) * 4).expect("perm"),
            Buffer::zeroed(&device, u64::from(sorted) * 4).expect("row experts"),
            Buffer::zeroed(&device, u64::from(sorted) * 4).expect("tile experts"),
            Buffer::zeroed(&device, u64::from(pairs) * 4).expect("inverse"),
            Buffer::zeroed(&device, u64::from(sorted) * u64::from(k) * 2).expect("sorted x"),
            Buffer::zeroed(&device, u64::from(sorted) * u64::from(n) * 2).expect("sorted y"),
            Buffer::zeroed(&device, u64::from(sorted) * u64::from(groups) * 4).expect("sums"),
        ];
        let scratch = RoutedScratch {
            perm: Tensor::new(bind(&scratch_b[0]), 1, sorted, Dtype::I32),
            row_expert: Tensor::new(bind(&scratch_b[1]), 1, sorted, Dtype::I32),
            tile_expert: Tensor::new(bind(&scratch_b[2]), 1, sorted, Dtype::I32),
            inv: Tensor::new(bind(&scratch_b[3]), 1, pairs, Dtype::I32),
            x: Tensor::new(bind(&scratch_b[4]), sorted, k, Dtype::Bf16),
            y: Tensor::new(bind(&scratch_b[5]), sorted, n, Dtype::Bf16),
            sums: Tensor::new(bind(&scratch_b[6]), sorted, groups, Dtype::F32),
        };
        for distinct in [64u32, 40, 24] {
            // each layer's routes: `distinct` experts of its own, each token's eight distinct
            let mut route_b = Vec::new();
            for l in 0..layers {
                let base = l * 61;
                let routes: Vec<u8> = (0..tokens)
                    .flat_map(|t| {
                        (0..top_k).flat_map(move |s| {
                            let local = (t * top_k + s) % distinct;
                            let local = if distinct >= top_k { (local / top_k) * top_k + (local + t) % top_k } else { local };
                            (((base + local) % experts) as i32).to_le_bytes()
                        })
                    })
                    .collect();
                route_b.push(upload(&routes));
            }
            let routes: Vec<Tensor> = route_b
                .iter()
                .map(|b| Tensor::new(bind(b), tokens, top_k, Dtype::I32))
                .collect();
            let mut line = format!("K={k:>4} N={n:>4} distinct {distinct:>2}:");
            for (name, overrides) in &tactics {
                let tuning = engine_metal::chip::tuning(overrides, false);
                let mut times = Vec::new();
                let mut note = "";
                for i in 0..12 {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    for r in &routes {
                        let ran = if *name == "qmv" {
                            moe::matmul_select_quant(&sink, x, bank, *r, y).map(|()| true)
                        } else {
                            moe::matmul_select_batched(
                                &sink,
                                "price",
                                x,
                                bank,
                                None,
                                *r,
                                experts,
                                scratch,
                                y,
                                &tuning,
                                moe::Sorted::default(),
                            )
                        };
                        match ran {
                            Ok(true) => {}
                            Ok(false) => note = " (unbatched)",
                            Err(_) => note = " (refused)",
                        }
                    }
                    let s = frame.commit_timed().expect("commit");
                    if i >= 2 {
                        times.push(s * 1e6 / f64::from(layers));
                    }
                }
                times.sort_by(f64::total_cmp);
                let us = times[times.len() / 2];
                line += &format!("  {name} {us:.1}us{note}");
            }
            eprintln!("{line}");
        }
    }
}
