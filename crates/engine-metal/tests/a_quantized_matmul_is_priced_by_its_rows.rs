#![cfg(target_vendor = "apple")]

use std::time::Instant;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

const QMM_FILE: &str = "linear/quant_qmm_t.metal";

fn split_block(m: u32) -> (u32, u32) {
    let bm = quant::bm_rung(i32::try_from(m).expect("rows fit")).unsigned_abs();
    (bm, m.div_ceil(bm) * bm)
}

const GROUP: u32 = 64;
const BITS: u32 = 4;

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

fn to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits((u32::from(hi) << 24) | (u32::from(lo) << 16))
}

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn list(name: &str, fallback: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn shapes() -> Vec<(u32, u32)> {
    std::env::var("PIE_QMM_SHAPES")
        .unwrap_or_else(|_| "5120x5120,5120x17408".to_string())
        .split(',')
        .filter_map(|s| {
            let (k, n) = s.trim().split_once('x')?;
            Some((k.trim().parse().ok()?, n.trim().parse().ok()?))
        })
        .collect()
}

#[test]
fn a_quantized_matmul_is_priced_by_its_rows_every_case() {
    every_row_count_is_timed();
    every_folded_point_answers_the_one_row_point();
}

fn every_row_count_is_timed() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    if let Ok(tuning) = std::env::var("PIE_QMM_TUNING") {
        let mut over = kernels_metal::tuning::Overrides::default();
        for pair in tuning.split(',').filter(|p| !p.trim().is_empty()) {
            let (key, value) = pair.split_once('=').expect("key=value");
            let value: u32 = value.trim().parse().expect("an integer knob");
            match key.trim() {
                "qmv_rows_packs" => over.qmv_rows_packs = Some(value),
                "qmv_rows_max" => over.qmv_rows_max = Some(value),
                "qmm_min_batch" => over.qmm_min_batch = Some(value),
                "qmm_min_batch_moe" => over.qmm_min_batch_moe = Some(value),
                "qmm_min_batch_emulated" => over.qmm_min_batch_emulated = Some(value),
                "fp16_qmm" => over.fp16_qmm = Some(value != 0),
                "qmm_bn_crossover_tg" => over.qmm_bn_crossover_tg = Some(value),
                other => panic!("no knob named {other} here"),
            }
        }
        assert!(
            kernels_metal::tuning::override_with(over),
            "the tuning is laid once"
        );
        eprintln!("tuning: {tuning}");
    }

    let rows = list("PIE_QMM_ROWS", "1,2,3,4,6,8,16");
    let steps: usize = env("PIE_QMM_STEPS", 50usize);
    let batch: usize = env("PIE_QMM_BATCH", 32usize);
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    let bits: u32 = env("PIE_QMM_BITS", BITS);
    let group: u32 = env("PIE_QMM_GROUP", GROUP);
    for (k, n) in shapes() {
        let words = u64::from(n) * u64::from(k) * u64::from(bits) / 32;
        let factors_n = u64::from(n) * u64::from(k / group);
        let widest = *rows.iter().max().expect("a row count");
        let mut codes_b = Buffer::zeroed(&device, words * 4).expect("codes");
        let mut scales_b = Buffer::zeroed(&device, factors_n * 2).expect("scales");
        let mut biases_b = Buffer::zeroed(&device, factors_n * 2).expect("biases");
        let cap = rows
            .iter()
            .map(|&m| split_block(m).1)
            .max()
            .expect("a row count")
            .max(widest);
        let split: u32 = env("PIE_QMM_SPLITK", 0u32);
        let precast_on: u32 = env("PIE_QMM_PRECAST", 1u32);
        let mut act_b = Buffer::zeroed(&device, u64::from(cap) * u64::from(k) * 2).expect("act");
        let out_b = Buffer::zeroed(&device, u64::from(cap) * u64::from(n) * 2).expect("out");
        let precast_b =
            Buffer::zeroed(&device, u64::from(cap) * u64::from(k) * 2).expect("precast");
        let partial_b = Buffer::zeroed(
            &device,
            u64::from(split.max(8)) * u64::from(cap.max(8)) * u64::from(n) * 4,
        )
        .expect("partials");
        {
            let mut codes = vec![0u8; usize::try_from(words * 4).expect("codes fit")];
            for (at, byte) in codes.iter_mut().enumerate() {
                *byte = noise(at as u64);
            }
            codes_b.write(0, &codes).expect("write codes");
            let mut factors = vec![0u8; usize::try_from(factors_n * 2).expect("factors fit")];
            for (at, pair) in factors.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let v = 0.01 + 0.001 * f32::from(noise(at as u64 ^ 0xAA) % 8);
                pair.copy_from_slice(&bf16(v));
            }
            scales_b.write(0, &factors).expect("write scales");
            let mut zeros = vec![0u8; usize::try_from(factors_n * 2).expect("factors fit")];
            for (at, pair) in zeros.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                pair.copy_from_slice(&bf16(-0.05 + 0.01 * f32::from(noise(at as u64 ^ 0x55) % 8)));
            }
            biases_b.write(0, &zeros).expect("write biases");
            let mut act =
                vec![0u8; usize::try_from(u64::from(cap) * u64::from(k) * 2).expect("act fits")];
            for (row, chunk) in act
                .chunks_exact_mut(usize::try_from(k).expect("k fits") * 2)
                .enumerate()
            {
                for (at, pair) in chunk.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let v = if row == 0 {
                        0.02 * (f32::from(noise(at as u64) % 16) - 8.0)
                    } else {
                        0.02 * (f32::from(noise(at as u64 ^ (row as u64) << 8) % 16) - 8.0)
                    };
                    pair.copy_from_slice(&bf16(v));
                }
            }
            act_b.write(0, &act).expect("write act");
        }
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
        let (hc, hs, hb, ha, ho) = (
            bind(&codes_b),
            bind(&scales_b),
            bind(&biases_b),
            bind(&act_b),
            bind(&out_b),
        );
        let (hp, hq) = (bind(&precast_b), bind(&partial_b));
        if split > 0 {
            eprintln!("arm: split-K x{split}, f32 partials, at the engine's row rung");
        } else if precast_on == 0 {
            eprintln!("arm: the ladder, precast plane withheld");
        } else {
            eprintln!("arm: the ladder, with a precast plane (the engine's rung for gs=64/b=4)");
        }

        eprintln!(
            "\n  K={k} N={n}  ({:.2} GiB of codes)",
            words as f64 * 4.0 / (1u64 << 30) as f64
        );
        let tol: f64 = env("PIE_QMM_TOL", 0.05f64);
        let mut one = 0.0f64;
        let mut reference: Option<Vec<f32>> = None;
        for &m in &rows {
            let bank = Bank {
                codes: Tensor::new(hc, n, k, Dtype::U4g64),
                scales: Tensor::new(hs, n, k / group, Dtype::Bf16),
                biases: Some(Tensor::new(hb, n, k / group, Dtype::Bf16)),
                group,
                bits,
            };
            let act = Tensor::new(ha, m, k, Dtype::Bf16);
            let y = Tensor::new(ho, m, n, Dtype::Bf16);
            let none = |_: u32, _: u32| None;
            let some = |rows: u32, contraction: u32| {
                (u64::from(rows) * u64::from(contraction) <= u64::from(cap) * u64::from(k))
                    .then(|| Tensor::new(hp, rows, contraction, Dtype::F16))
            };
            let precast: &dyn Fn(u32, u32) -> Option<Tensor> =
                if precast_on == 0 { &none } else { &some };
            let partial_rows = partial_b.bytes() / (u64::from(n) * 4);
            let some_partials = |rows: u32, width: u32| {
                (width == n && u64::from(rows) <= partial_rows)
                    .then(|| Tensor::new(hq, rows, width, Dtype::F32))
            };
            let partials: &dyn Fn(u32, u32) -> Option<Tensor> =
                if env("PIE_QMM_LADDER_SPLIT", 1u32) == 0 {
                    &none
                } else {
                    &some_partials
                };
            let launch = |sink: &dyn Encode| {
                if split == 0 {
                    let scratch = quant::Scratch { precast, partials };
                    return quant::matmul(sink, act, bank, y, scratch, cap).expect("the launch");
                }
                let (bm, padded) = split_block(m);
                let (bm_i, padded_i, n_i, k_i, split_i) = (
                    i32::try_from(bm).expect("bm"),
                    i32::try_from(padded).expect("padded"),
                    i32::try_from(n).expect("n"),
                    i32::try_from(k).expect("k"),
                    i32::try_from(split).expect("split"),
                );
                assert!(
                    k_i % (split_i * 32) == 0 && (k_i / split_i) % 64 == 0,
                    "K={k} must split into {split} partitions of whole 32-blocks and 64-groups"
                );
                let entry =
                    format!("affine_qmm_t_splitk_f32_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_32");
                let entry: &'static str = Box::leak(entry.into_boxed_str());
                let grid =
                    quant::qmm_grid("bench", n_i, 32, padded_i, bm_i, split_i).expect("a grid");
                let stride = padded_i * n_i;
                sink.fire(
                    Fire::at(QMM_FILE, entry).apply(Grid::of(grid, quant::qmm_group(bm_i))),
                    &[
                        Tensor::new(hc, n, k, Dtype::U4g64).arg(),
                        Tensor::new(hs, n, k / group, Dtype::Bf16).arg(),
                        Tensor::new(hb, n, k / group, Dtype::Bf16).arg(),
                        act.arg(),
                        sink.absent().expect("a null seat"),
                        k_i.arg(),
                        n_i.arg(),
                        sink.absent().expect("a null seat"),
                        Tensor::new(hq, padded * split, n, Dtype::F32).arg_mut(),
                        (k_i / split_i).arg(),
                        stride.arg(),
                    ],
                )
                .expect("the split-K launch");
                let mut reduce = vec![sink.absent().expect("a null seat"); 4];
                reduce.push(y.arg_mut());
                reduce.push(sink.absent().expect("a null seat"));
                reduce.push(n_i.arg());
                reduce.push(sink.absent().expect("a null seat"));
                reduce.push(Tensor::new(hq, padded * split, n, Dtype::F32).arg());
                reduce.push(sink.absent().expect("a null seat"));
                reduce.push(stride.arg());
                reduce.push(split_i.arg());
                sink.fire(
                    Fire::at(QMM_FILE, "qmm_splitk_reduce_f32_bfloat16")
                        .apply(Grid::of([n, m, 1], [256, 1, 1])),
                    &reduce,
                )
                .expect("the reduce");
            };
            {
                let warm_ms: u64 = env("PIE_QMM_WARM_MS", 300u64);
                let began = Instant::now();
                loop {
                    let frame = device.frame().expect("a frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    for _ in 0..batch {
                        launch(&sink);
                    }
                    frame.commit().expect("the warm commit");
                    if began.elapsed().as_millis() as u64 >= warm_ms {
                        break;
                    }
                }
            }

            let began = Instant::now();
            let mut device_s = 0.0f64;
            for _ in 0..steps {
                let frame = device.frame().expect("a frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for _ in 0..batch {
                    launch(&sink);
                }
                device_s += frame.commit_timed().expect("the commit");
            }
            let launches = (steps * batch) as f64;
            let raw = handles.read(ho, u64::from(n) * 2).expect("read row 0");
            let got: Vec<f32> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| to_f32(p[0], p[1]))
                .collect();
            match &reference {
                None => reference = Some(got),
                Some(want) => {
                    let mut worst = 0.0f64;
                    let mut at = 0usize;
                    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                        let scale = f64::from(a.abs()).max(f64::from(b.abs())).max(0.1);
                        let d = f64::from((a - b).abs()) / scale;
                        if d > worst {
                            worst = d;
                            at = i;
                        }
                    }
                    if worst > tol && std::env::var_os("PIE_QMM_DEBUG").is_some() {
                        let off: Vec<usize> = want
                            .iter()
                            .zip(&got)
                            .enumerate()
                            .filter(|(_, (a, b))| {
                                let (a, b) = (**a, **b);
                                let scale = f64::from(a.abs()).max(f64::from(b.abs())).max(0.1);
                                f64::from((a - b).abs()) / scale > tol
                            })
                            .map(|(i, _)| i)
                            .collect();
                        eprintln!(
                            "    {} of {} columns off; first 24: {:?}",
                            off.len(),
                            want.len(),
                            &off[..off.len().min(24)]
                        );
                        eprintln!(
                            "    col%8 histogram: {:?}",
                            (0..8)
                                .map(|k| off.iter().filter(|&&i| i % 8 == k).count())
                                .collect::<Vec<_>>()
                        );
                    }
                    assert!(
                        worst <= tol,
                        "rows {m} answers row 0 differently from rows {}: worst relative {worst:.4} at column {at} ({} vs {}); a point that disagrees here is wrong before it is slow",
                        rows[0],
                        want[at],
                        got[at]
                    );
                }
            }

            let wall_us = began.elapsed().as_secs_f64() * 1e6 / launches;
            let dev_us = device_s * 1e6 / launches;
            if m == rows[0] {
                one = dev_us;
            }
            let gb = words as f64 * 4.0 / 1e9;
            eprintln!(
                "    rows {m:>3}: {dev_us:>8.1} us device ({:>5.2}x one row, {:>6.1} us/row)  {:>6.1} GB/s  wall {wall_us:>8.1} us",
                dev_us / one.max(1e-9),
                dev_us / f64::from(m),
                gb / (dev_us / 1e6),
            );
        }
    }
}

fn every_folded_point_answers_the_one_row_point() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let (k, n): (u32, u32) = (2048, 256);
    let rows_max = 8u32;
    let codes_bytes = u64::from(n) * u64::from(k);
    let factors_max = u64::from(n) * u64::from(k / 32);
    let mut codes_b = Buffer::zeroed(&device, codes_bytes).expect("codes");
    let mut scales_b = Buffer::zeroed(&device, factors_max * 2).expect("scales");
    let mut biases_b = Buffer::zeroed(&device, factors_max * 2).expect("biases");
    let mut act_b = Buffer::zeroed(&device, u64::from(rows_max) * u64::from(k) * 2).expect("act");
    let fold_b = Buffer::zeroed(&device, u64::from(rows_max) * u64::from(n) * 2).expect("fold out");
    let one_b = Buffer::zeroed(&device, u64::from(rows_max) * u64::from(n) * 2).expect("one out");
    {
        let mut codes = vec![0u8; usize::try_from(codes_bytes).expect("fits")];
        for (at, byte) in codes.iter_mut().enumerate() {
            *byte = noise(at as u64);
        }
        codes_b.write(0, &codes).expect("write codes");
        let mut factors = vec![0u8; usize::try_from(factors_max * 2).expect("fits")];
        for (at, pair) in factors.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(0.01 + 0.001 * f32::from(noise(at as u64 ^ 0xAA) % 8)));
        }
        scales_b.write(0, &factors).expect("write scales");
        for (at, pair) in factors.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(-0.05 + 0.01 * f32::from(noise(at as u64 ^ 0x55) % 8)));
        }
        biases_b.write(0, &factors).expect("write biases");
        let mut act =
            vec![0u8; usize::try_from(u64::from(rows_max) * u64::from(k) * 2).expect("fits")];
        for (at, pair) in act.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(0.02 * (f32::from(noise(at as u64) % 16) - 8.0)));
        }
        act_b.write(0, &act).expect("write act");
    }
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hc, hs, hb, ha, hf, ho) = (
        bind(&codes_b),
        bind(&scales_b),
        bind(&biases_b),
        bind(&act_b),
        bind(&fold_b),
        bind(&one_b),
    );
    let read = |h: u32, rows: u32| -> Vec<f32> {
        handles
            .read(h, u64::from(rows) * u64::from(n) * 2)
            .expect("read")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| to_f32(p[0], p[1]))
            .collect()
    };

    let mut swept = 0usize;
    let mut wrong = Vec::new();
    for &gs in &[32i32, 64, 128] {
        for &bits in &[2i32, 4, 8] {
            for &packs in &[1i32, 2] {
                for rung in 2..=8i32 {
                    let Ok(point) = quant::qmv_rows_point("sweep", gs, bits, rung, packs) else {
                        continue;
                    };
                    let m = rung.unsigned_abs();
                    let (ki, ni, mi) = (i32::try_from(k).unwrap(), i32::try_from(n).unwrap(), rung);
                    let frame = device.frame().expect("a frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    let w = [
                        Tensor::new(hc, n, k, Dtype::U4g64).arg(),
                        Tensor::new(hs, n, 1, Dtype::Bf16).arg(),
                        Tensor::new(hb, n, 1, Dtype::Bf16).arg(),
                        Tensor::new(ha, m, k, Dtype::Bf16).arg(),
                    ];
                    let mut fold = w.to_vec();
                    fold.extend([
                        Tensor::new(hf, m, n, Dtype::Bf16).arg_mut(),
                        ki.arg(),
                        ni.arg(),
                        mi.arg(),
                    ]);
                    sink.fire(
                        Fire::at("linear/quant_qmv_rows.metal", point.entry)
                            .stamp(point.stamp)
                            .apply(Grid::of(
                                quant::qmv_rows_grid("sweep", mi, rung, ni).expect("grid"),
                                [32, 2, 1],
                            )),
                        &fold,
                    )
                    .expect("the fold");
                    let one =
                        quant::qmv_point("sweep", "fast", gs, bits).expect("the one-row point");
                    let mut single = w.to_vec();
                    single.extend([
                        Tensor::new(ho, m, n, Dtype::Bf16).arg_mut(),
                        ki.arg(),
                        ni.arg(),
                    ]);
                    sink.fire(
                        Fire::at("linear/quant_qmv.metal", one.entry).apply(Grid::of(
                            quant::qmv_grid("sweep", mi, ni).expect("grid"),
                            [32, 2, 1],
                        )),
                        &single,
                    )
                    .expect("the one-row point");
                    frame.commit().expect("commit");
                    let (got, want) = (read(hf, m), read(ho, m));
                    let mut worst = 0.0f64;
                    for (a, b) in want.iter().zip(&got) {
                        let scale = f64::from(a.abs()).max(f64::from(b.abs())).max(1e-3);
                        worst = worst.max(f64::from((a - b).abs()) / scale);
                    }
                    let tol = if packs == 2 { 0.0 } else { 0.02 };
                    swept += 1;
                    if worst > tol {
                        wrong.push(format!("{} (worst relative {worst:.4})", point.entry));
                    }
                }
            }
        }
    }
    eprintln!("swept {swept} folded points against the one-row point");
    assert!(
        wrong.is_empty(),
        "{} folded points answer differently from the one-row point:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}
