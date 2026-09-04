//! **WHAT ONE QUANTIZED PROJECTION COSTS AS ITS ROW COUNT GROWS** — the
//! kernel underneath `a_fire_is_priced_by_its_width`, timed on its own so an
//! experiment is seconds instead of a two-minute model run.
//!
//! A decode fire is weight-bound: the bank is read once and spent on `M`
//! activation rows, so the ideal curve is flat in `M` until arithmetic
//! catches up. It is not flat. Two families cover the axis and neither
//! covers the middle — the vector point holds each row's accumulators in
//! REGISTERS (`quant_qmv_rows.metal` folds at most `qmv_rows_max` rows), and
//! the tile point is built on `mlx::steel::BlockMMA`, whose `kFragSize = 8`
//! forces `BM % 8 == 0`. So `M` in 3..7 falls between a register ceiling and
//! a fragment floor, and this bench is where that shows up as a number.
//!
//! Asserts nothing; the numbers are the finding. One bank per shape, filled
//! with arbitrary codes — the arithmetic is the same whatever the bits are,
//! and nothing here reads the product.
//!
//! ```text
//! PIE_QMM_SHAPES=5120x5120,5120x17408 PIE_QMM_ROWS=1,2,3,4,6,8,16 \
//!   [PIE_QMM_TUNING=qmv_rows_max=4] [PIE_QMM_STEPS=50] \
//!   cargo test -p engine-metal --release --test a_quantized_matmul_is_priced_by_its_rows -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::time::Instant;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

/// Codes per scale entry and bits per code — the shape every 4-bit MLX
/// conversion in this tree ships.
const GROUP: u32 = 64;
const BITS: u32 = 4;

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn list(name: &str, fallback: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// `(K, N)` pairs: the contraction and the output width of one projection.
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
fn every_row_count_is_timed() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    // The same knobs `a_fire_is_priced_by_its_width` lays, so a curve
    // measured here and a fire measured there are the same dispatch.
    if let Ok(tuning) = std::env::var("PIE_QMM_TUNING") {
        let mut over = kernels_metal::tuning::Overrides::default();
        for pair in tuning.split(',').filter(|p| !p.trim().is_empty()) {
            let (key, value) = pair.split_once('=').expect("key=value");
            let value: u32 = value.trim().parse().expect("an integer knob");
            match key.trim() {
                "qmv_rows_packs" => over.qmv_rows_packs = Some(value),
                "qmv_rows_max" => over.qmv_rows_max = Some(value),
                "qmm_min_batch" => over.qmm_min_batch = Some(value),
                "qmm_bn_crossover_tg" => over.qmm_bn_crossover_tg = Some(value),
                other => panic!("no knob named {other} here"),
            }
        }
        assert!(kernels_metal::tuning::override_with(over), "the tuning is laid once");
        eprintln!("tuning: {tuning}");
    }

    let rows = list("PIE_QMM_ROWS", "1,2,3,4,6,8,16");
    let steps: usize = env("PIE_QMM_STEPS", 50usize);
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    for (k, n) in shapes() {
        // One bank: codes packed `BITS` apiece into u32 words, one bf16
        // scale and zero point per `GROUP` codes.
        let words = u64::from(n) * u64::from(k) * u64::from(BITS) / 32;
        let factors = u64::from(n) * u64::from(k / GROUP);
        let widest = *rows.iter().max().expect("a row count");
        let codes_b = Buffer::zeroed(&device, words * 4).expect("codes");
        let scales_b = Buffer::zeroed(&device, factors * 2).expect("scales");
        let biases_b = Buffer::zeroed(&device, factors * 2).expect("biases");
        let act_b = Buffer::zeroed(&device, u64::from(widest) * u64::from(k) * 2).expect("act");
        let out_b = Buffer::zeroed(&device, u64::from(widest) * u64::from(n) * 2).expect("out");
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
        let (hc, hs, hb, ha, ho) =
            (bind(&codes_b), bind(&scales_b), bind(&biases_b), bind(&act_b), bind(&out_b));

        eprintln!("\n  K={k} N={n}  ({:.2} GiB of codes)", words as f64 * 4.0 / (1u64 << 30) as f64);
        let mut one = 0.0f64;
        for &m in &rows {
            let bank = Bank {
                // The codes tensor is stated in CODES, not words: the point
                // derives the packing from `bits`.
                codes: Tensor::new(hc, n, k, Dtype::U4g64),
                scales: Tensor::new(hs, n, k / GROUP, Dtype::Bf16),
                biases: Some(Tensor::new(hb, n, k / GROUP, Dtype::Bf16)),
                group: GROUP,
                bits: BITS,
            };
            let act = Tensor::new(ha, m, k, Dtype::Bf16);
            let y = Tensor::new(ho, m, n, Dtype::Bf16);
            let none = |_: u32, _: u32| None;
            let scratch = quant::Scratch { precast: &none };
            // `capacity_rows` is the ROW CAPACITY of the output rectangle, not
            // this launch's row count: `mb_block` pads `M` up to a `BM` rung
            // and refuses a padding the capacity cannot hold. Passing `m`
            // here would deny the tile point every row count it must pad
            // (M = 6 would fall to three folds of two), which is not what a
            // fire with a wider rectangle does.
            let cap = widest;
            let _ = cap;

            // Warm: the first launch compiles the point.
            {
                let frame = device.frame().expect("a frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                quant::matmul(&sink, act, bank, y, scratch, widest).expect("the warm launch");
                frame.commit().expect("the warm commit");
            }

            let began = Instant::now();
            let mut device_s = 0.0f64;
            for _ in 0..steps {
                let frame = device.frame().expect("a frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                let scratch = quant::Scratch { precast: &none };
                quant::matmul(&sink, act, bank, y, scratch, widest).expect("the launch");
                device_s += frame.commit_timed().expect("the commit");
            }
            let wall_us = began.elapsed().as_secs_f64() * 1e6 / steps as f64;
            let dev_us = device_s * 1e6 / steps as f64;
            if m == rows[0] {
                one = dev_us;
            }
            // Bytes the bank alone costs, against the device time: a
            // weight-bound point should sit near the machine's bandwidth
            // and stay there as rows grow.
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
