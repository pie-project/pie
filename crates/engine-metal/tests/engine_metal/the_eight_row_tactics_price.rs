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
        ("reg-packed", "[kernel]\nqmm = [\"register\", \"qmv\"]"),
        ("reg-s1", "[kernel]\nqmm = [\"register\", \"qmv\"]"),
        ("reg-s2", "[kernel]\nqmm = [\"register\", \"qmv\"]"),
        ("reg-s4", "[kernel]\nqmm = [\"register\", \"qmv\"]"),
        ("reg-s8", "[kernel]\nqmm = [\"register\", \"qmv\"]"),
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
            // each bank's codes and factors as the MPP pack lays them, for the packed register walk
            let mut packed_banks = Vec::new();
            let mut packed_planes = Vec::new();
            for bank in &banks {
                let words = n * (k / 8);
                let bytes = u64::from(n) * u64::from(k) / 2 + u64::from(n) * u64::from(k / 64) * 4;
                let buffer = Buffer::zeroed(&device, bytes).expect("packed");
                let t = Tensor::new(bind(&buffer), n, k, Dtype::U4g64);
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                use kernels_metal::encode::{Arg, Encode};
                sink.fire(
                    kernels_metal::encode::Fire::at("linear/quant_qmm_mpp.metal", "qmm_mpp_pack_bank")
                        .apply(kernels_metal::encode::Grid::of([words, 1, 1], [256, 1, 1])),
                    &[
                        bank.codes.arg(),
                        bank.scales.arg(),
                        bank.biases.unwrap().arg(),
                        t.arg_mut(),
                        k.arg(),
                        n.arg(),
                    ],
                )
                .expect("pack");
                frame.commit().expect("commit");
                let mut b = *bank;
                b.mpp_codes = Some(t);
                packed_banks.push(b);
                packed_planes.push(buffer);
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
            let mut reference = Vec::new();
            for (name, overrides) in &tactics {
                let tuning = engine_metal::chip::tuning(overrides, false);
                let use_packed = *name == "reg-packed";
                if use_packed {
                    unsafe { std::env::set_var("PIE_SG_PACKED", "1") };
                } else {
                    unsafe { std::env::remove_var("PIE_SG_PACKED") };
                }
                let banks_now = if use_packed { &packed_banks } else { &banks };
                match name.strip_prefix("reg-s") {
                    Some(sp) => unsafe { std::env::set_var("PIE_SG_SPLIT", sp) },
                    None => unsafe { std::env::remove_var("PIE_SG_SPLIT") },
                }
                let mut times = Vec::new();
                let mut failed = false;
                for i in 0..12 {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    for bank in banks_now {
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
                if *name == "register" || *name == "reg-packed" {
                    let got = handles.read(ho, u64::from(m * n * 2)).expect("read");
                    if *name == "register" {
                        reference = got;
                    } else {
                        assert!(got == reference, "the packed walk lands other bits at K={k} N={n} M={m}");
                    }
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
    let tactics: [(&str, &str); 8] = [
        ("qmv", ""),
        ("grouped", ""),
        ("ranked", ""),
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
            let mut outs = Vec::new();
            for variant in 0..3 {
                unsafe { std::env::remove_var("PIE_GROUPED") };
                unsafe { std::env::remove_var("PIE_RANKED") };
                if variant == 1 {
                    unsafe { std::env::set_var("PIE_GROUPED", "1") };
                } else if variant == 2 {
                    unsafe { std::env::set_var("PIE_RANKED", "1") };
                }
                let mut got = Vec::new();
                for r in &routes[..3] {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    moe::matmul_select_quant(&sink, x, bank, *r, y).expect("routed");
                    frame.commit().expect("commit");
                    got.push(handles.read(y.buf, u64::from(pairs * n * 2)).expect("read"));
                }
                outs.push(got);
            }
            assert!(outs[0] == outs[1], "the grouped kernel lands other bits at K={k} distinct {distinct}");
            assert!(outs[0] == outs[2], "the ranked kernel lands other bits at K={k} distinct {distinct}");
            let mut line = format!("K={k:>4} N={n:>4} distinct {distinct:>2}:");
            for (name, overrides) in &tactics {
                let tuning = engine_metal::chip::tuning(overrides, false);
                let mut times = Vec::new();
                let mut note = "";
                for i in 0..12 {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    for r in &routes {
                        if *name == "grouped" {
                            unsafe { std::env::set_var("PIE_GROUPED", "1") };
                        } else {
                            unsafe { std::env::remove_var("PIE_GROUPED") };
                        }
                        if *name == "ranked" {
                            unsafe { std::env::set_var("PIE_RANKED", "1") };
                        } else {
                            unsafe { std::env::remove_var("PIE_RANKED") };
                        }
                        let ran = if *name == "qmv" || *name == "grouped" || *name == "ranked" {
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

/// The 35B's GDN in-projection as the engine launches it (its 64 gate columns,
/// and its 12288 projection columns landed in halves of 8192 and 4096, in one
/// segment launch) against the plain register walk of the 12288 columns.
#[test]
#[ignore = "a microbenchmark"]
fn the_eight_row_segments_price() {
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
    let (k, m) = (2048u32, 8u32);
    let groups = k / 64;
    let make = |n: u32, seed: u64, keep: &mut Vec<[Buffer; 3]>| {
        let codes: Vec<u8> = (0..u64::from(n) * u64::from(k) / 2).map(|at| noise(at ^ seed)).collect();
        let scales: Vec<u8> = (0..n * groups)
            .flat_map(|at| bf16(0.0031 + f32::from(noise(u64::from(at) ^ seed ^ 0x11)) * 0.000173))
            .collect();
        let biases: Vec<u8> = (0..n * groups)
            .flat_map(|at| bf16(-0.2 + f32::from(noise(u64::from(at) ^ seed ^ 0x5678)) * 0.00157))
            .collect();
        let p = [upload(&codes), upload(&scales), upload(&biases)];
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(bind(&p[0]), n, k, Dtype::U4g64),
            scales: Tensor::new(bind(&p[1]), n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&p[2]), n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        keep.push(p);
        bank
    };
    let mut keep = Vec::new();
    let copies = 22;
    let pairs: Vec<(Bank, Bank)> = (0..copies)
        .map(|c| (make(64, c as u64 * 77, &mut keep), make(12288, c as u64 * 77 + 1, &mut keep)))
        .collect();
    let inputs: Vec<u8> = (0..m * k)
        .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0xabcd)) - 127.0) * 0.0197))
        .collect();
    let act_b = upload(&inputs);
    let act = Tensor::new(bind(&act_b), m, k, Dtype::Bf16);
    let outs: Vec<Buffer> = [64u32, 4096, 8192, 12288]
        .iter()
        .map(|&n| upload(&vec![0u8; (m * n * 2) as usize]))
        .collect();
    let y = |i: usize, n: u32| Tensor::new(bind(&outs[i]), m, n, Dtype::Bf16);
    let table_rows = (17 * k + 4 * 256).div_ceil(2 * k) + 1;
    let table_b = upload(&vec![0u8; (table_rows * k * 2) as usize]);
    let table = Tensor::new(bind(&table_b), table_rows, k, Dtype::Bf16);
    let partial_b = upload(&vec![0u8; (8 * m * 12288 * 4 + 8 * m * 64 * 4) as usize]);
    let partials = Tensor::new(bind(&partial_b), 1, 8 * m * (12288 + 64), Dtype::F32);
    let precast = |rows: u32, width: u32| {
        (rows <= table_rows && width == k).then(|| Tensor::new(bind(&table_b), rows, width, Dtype::F16))
    };
    let pb = upload(&vec![0u8; (8 * m * 12288 * 4) as usize + 64]);
    let partials_plain = |rows: u32, width: u32| {
        (rows <= 64 && width <= 12288).then(|| Tensor::new(bind(&pb), rows, width, Dtype::F32))
    };
    let tuning = engine_metal::chip::tuning("[kernel]\nqmm = [\"register\", \"qmv\"]", false);
    let cores = tuning.cores();
    let split = |n: u32| quant::register_split(n as i32, k as i32, cores);
    for round in 0..2 {
        let mut seg_t = Vec::new();
        let mut plain_t = Vec::new();
        for i in 0..12 {
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            for (a, b) in &pairs {
                let mut segments = vec![
                    quant::Segment { w: *a, first: 0, y: y(0, 64), split: split(64) },
                    quant::Segment { w: *b, first: 8192, y: y(1, 4096), split: split(12288) },
                    quant::Segment { w: *b, first: 0, y: y(2, 8192), split: split(12288) },
                ];
                segments.sort_by_key(|s| s.y.width);
                let (counters, _) = quant::segments_scratch(&segments, m);
                quant::register_table(&sink, act, table, m as i32, k as i32, counters as i32).expect("table");
                quant::register_segments(&sink, &segments, table, m as i32, k as i32, Some(partials))
                    .expect("segments");
            }
            let s = frame.commit_timed().expect("commit");
            if i >= 2 {
                seg_t.push(s * 1e6 / copies as f64);
            }
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            for (_, b) in &pairs {
                quant::matmul(
                    &sink,
                    &tuning,
                    act,
                    *b,
                    y(3, 12288),
                    quant::Scratch { precast: &precast, partials: &partials_plain, table: None },
                    8,
                )
                .expect("plain");
            }
            let s = frame.commit_timed().expect("commit");
            if i >= 2 {
                plain_t.push(s * 1e6 / copies as f64);
            }
        }
        seg_t.sort_by(f64::total_cmp);
        plain_t.sort_by(f64::total_cmp);
        eprintln!(
            "round {round}: GDN in-projection segments {:.1}us (splits {} {}), plain 12288 walk {:.1}us (table included in both)",
            seg_t[seg_t.len() / 2],
            split(64),
            split(12288),
            plain_t[plain_t.len() / 2]
        );
    }
}

/// The shared expert as a ninth route of each token (one expert every token
/// takes) in the routed vector launch, against the eight routes alone.
#[test]
#[ignore = "a microbenchmark"]
fn the_eight_row_shared_route_prices() {
    use kernels_metal::linear::moe;
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
    let (experts, tokens, layers) = (257u32, 8u32, 40u32);
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
        for slots in [8u32, 9] {
            let pairs = tokens * slots;
            let x_rows = if per_pair { pairs } else { tokens };
            let inputs: Vec<u8> = (0..x_rows * k)
                .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0xabcd)) - 127.0) * 0.0197))
                .collect();
            let x_b = upload(&inputs);
            let x = Tensor::new(bind(&x_b), x_rows, k, Dtype::Bf16);
            let y_b = upload(&vec![0u8; (pairs * n * 2) as usize]);
            let y = Tensor::new(bind(&y_b), pairs, n, Dtype::Bf16);
            let route_b: Vec<Buffer> = (0..layers)
                .map(|l| {
                    let routes: Vec<u8> = (0..tokens)
                        .flat_map(|t| {
                            (0..slots).flat_map(move |s| {
                                let e = if s == 8 { 256 } else { (l * 61 + t * 5 + s * 29) % 256 };
                                (e as i32).to_le_bytes()
                            })
                        })
                        .collect();
                    upload(&routes)
                })
                .collect();
            let routes: Vec<Tensor> = route_b
                .iter()
                .map(|b| Tensor::new(bind(b), tokens, slots, Dtype::I32))
                .collect();
            let mut times = Vec::new();
            for i in 0..12 {
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for r in &routes {
                    moe::matmul_select_quant(&sink, x, bank, *r, y).expect("routed");
                }
                let s = frame.commit_timed().expect("commit");
                if i >= 2 {
                    times.push(s * 1e6 / f64::from(layers));
                }
            }
            times.sort_by(f64::total_cmp);
            eprintln!("K={k:>4} N={n:>4} {slots} routes a token: {:.1}us", times[times.len() / 2]);
        }
    }
}
