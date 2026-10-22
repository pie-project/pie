#![cfg(target_vendor = "apple")]
//! Microbench (not a test): the Apple10 register-fed (cooperative) tile over the dense
//! projections of a Qwen3.8-27B speculative verify (8 rows, u4g64 affine), each projection's
//! layers in distinct banks so the weights stream from memory as a fire's do. Per split, the
//! device time of one launch (with its reduction) and the weight bytes it streams a second, and
//! a hash of the outputs, so a change to the tile can be held to the same bits.
//! `PIE_COOP_BENCH_DIVISOR` divides every projection's columns, to stand a GPU of fewer cores
//! in for a wider one (4 on a 10-core M5 for a 40-core M5 Max).
//! Run: cargo test --release -p engine-metal --test engine_metal \
//!        the_coop_tiles_price_their_matmuls -- --ignored --nocapture

use std::cell::RefCell;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

const FRAMES: usize = 15;
const ROWS: u32 = 8;
/// The weight bytes each projection's layers hold together, past any cache.
const STREAM: u64 = 768 << 20;

fn fill(seed: u64, bytes: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..bytes.div_ceil(8))
        .flat_map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x.to_le_bytes()
        })
        .take(bytes as usize)
        .collect()
}

fn bf16(v: f32) -> [u8; 2] {
    let b = v.to_bits();
    [(b >> 16) as u8, (b >> 24) as u8]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

struct Bench<'a> {
    device: &'a Context,
    handles: &'a Handles,
    pipelines: &'a Pipelines,
    keep: RefCell<Vec<Buffer>>,
}

impl Bench<'_> {
    fn upload(&self, bytes: &[u8]) -> u32 {
        let mut b = Buffer::zeroed(self.device, bytes.len().max(16) as u64).expect("a plane");
        b.write(0, bytes).expect("upload");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("bind");
        self.keep.borrow_mut().push(b);
        h
    }

    fn zeroed(&self, bytes: u64) -> u32 {
        let b = Buffer::zeroed(self.device, bytes.max(16)).expect("a plane");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("bind");
        self.keep.borrow_mut().push(b);
        h
    }

    fn bank(&self, seed: u64, k: u32, n: u32) -> Bank {
        let groups = k / 64;
        let scales: Vec<u8> = (0..n * groups)
            .flat_map(|a| bf16(0.003 + ((a ^ seed as u32) % 7) as f32 * 1e-4))
            .collect();
        let biases: Vec<u8> = (0..n * groups)
            .flat_map(|a| bf16((((a ^ seed as u32) % 9) as f32 - 4.0) / 64.0))
            .collect();
        Bank {
            mpp_codes: None,
            codes: Tensor::new(
                self.upload(&fill(seed, u64::from(k) * u64::from(n) / 2)),
                n,
                k,
                Dtype::U4g64,
            ),
            scales: Tensor::new(self.upload(&scales), n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(self.upload(&biases), n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        }
    }

    /// The device time of each frame `body` encodes, median over the timed frames, after one warm frame.
    fn time(&self, body: &dyn Fn(&Sink<'_>)) -> f64 {
        let mut v = Vec::new();
        for i in 0..=FRAMES {
            let frame = self.device.frame().expect("frame");
            let sink = Sink::new(self.device, &frame, self.pipelines, self.handles);
            body(&sink);
            let s = frame.commit_timed().expect("commit");
            if i > 0 {
                v.push(s * 1e6);
            }
        }
        median(v)
    }

    fn read(&self, t: Tensor) -> Vec<u8> {
        self.handles
            .read(t.buf, u64::from(t.rows) * u64::from(t.width) * 2)
            .expect("read back")
    }
}

#[test]
#[ignore = "a microbenchmark"]
fn the_coop_tiles_price_their_matmuls() {
    let Ok(device) = Context::bind() else { return };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let b = Bench {
        device: &device,
        handles: &handles,
        pipelines: &pipelines,
        keep: RefCell::new(Vec::new()),
    };
    let divisor: u32 = std::env::var("PIE_COOP_BENCH_DIVISOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let plane = Buffer::zeroed(&device, 64 << 20).expect("partials");
    let partials = Tensor::new(
        handles.bind(&plane, 0, plane.bytes()).expect("bind"),
        1,
        16 << 20,
        Dtype::F32,
    );
    // (what, K, N): the verify's projections
    for (what, k, n) in [
        ("down", 17408u32, 5120u32),
        ("gdn/attn out", 6144, 5120),
        ("gate_up", 5120, 34816),
        ("gdn in_proj", 5120, 16384),
        ("attn q+gate", 5120, 12288),
    ] {
        let n = n / divisor;
        let bytes = u64::from(k) * u64::from(n) * 9 / 16;
        let layers = STREAM.div_ceil(bytes).clamp(2, 64) as u32;
        let banks: Vec<Bank> = (0..layers)
            .map(|l| b.bank(u64::from(l) * 7919 + u64::from(n), k, n))
            .collect();
        let x: Vec<u8> = fill(u64::from(k) ^ 0xabcd, u64::from(ROWS) * u64::from(k) * 2)
            .chunks(2)
            .flat_map(|c| bf16((f32::from(c[0]) - 127.5) / 256.0))
            .collect();
        let act = Tensor::new(b.upload(&x), ROWS, k, Dtype::Bf16);
        let ys: Vec<Tensor> = (0..layers)
            .map(|_| Tensor::new(b.zeroed(u64::from(ROWS * n) * 2), ROWS, n, Dtype::Bf16))
            .collect();
        let mut line = format!("{what:13} K={k:5} N={n:6} x{layers:2}");
        for split in [1, 2, 4, 8] {
            if (k / 256) % split != 0 {
                continue;
            }
            let run = |ctx: &Sink<'_>| {
                for (w, y) in banks.iter().zip(&ys) {
                    quant::coop_matmul(
                        ctx,
                        "bench",
                        act,
                        *w,
                        w.biases.expect("biases"),
                        *y,
                        (ROWS as i32, n as i32, k as i32, n as i32, k as i32),
                        split as i32,
                        (split > 1).then_some(partials),
                    )
                    .expect("matmul");
                }
            };
            let us = b.time(&run) / f64::from(layers);
            let h = ys
                .iter()
                .fold(0u64, |h, y| h.rotate_left(5) ^ hash(&b.read(*y)));
            line += &format!(
                "  k{split} {us:7.1}us {:5.0}GB/s #{:04x}",
                bytes as f64 / us / 1e3,
                h & 0xffff
            );
        }
        eprintln!("{line}");
        b.keep.borrow_mut().clear();
    }
    // The gate and up of each layer then their swiglu, as two launches and as one.
    let tuning = engine_metal::chip::tuning("[kernel]\nqmm = [\"staged\", \"qmv\"]", false);
    // at least the columns the whole tile takes unsplit
    let (k, intermediate) = (5120u32, (17408 / divisor).max(6144));
    let bytes = u64::from(k) * u64::from(2 * intermediate) * 9 / 16;
    let layers = STREAM.div_ceil(bytes).clamp(2, 64) as u32;
    let banks: Vec<Bank> = (0..layers)
        .map(|l| b.bank(u64::from(l) * 7919 + 3, k, 2 * intermediate))
        .collect();
    let x: Vec<u8> = fill(0x5150, u64::from(ROWS) * u64::from(k) * 2)
        .chunks(2)
        .flat_map(|c| bf16((f32::from(c[0]) - 127.5) / 64.0))
        .collect();
    let act = Tensor::new(b.upload(&x), ROWS, k, Dtype::Bf16);
    let packed = Tensor::new(
        b.zeroed(u64::from(ROWS * 2 * intermediate) * 2),
        ROWS,
        2 * intermediate,
        Dtype::Bf16,
    );
    let hs: Vec<Tensor> = (0..layers)
        .map(|_| {
            Tensor::new(
                b.zeroed(u64::from(ROWS * intermediate) * 2),
                ROWS,
                intermediate,
                Dtype::Bf16,
            )
        })
        .collect();
    let mut line = format!("gate_up+swiglu K={k:5} I={intermediate:6} x{layers:2}");
    let mut reference: Option<Vec<Vec<u8>>> = None;
    for fused in [false, true] {
        let run = |ctx: &Sink<'_>| {
            for (w, h) in banks.iter().zip(&hs) {
                if fused {
                    assert!(
                        quant::coop_swiglu(ctx, &tuning, act, *w, intermediate, *h, 8)
                            .expect("fused"),
                        "the fused tile takes the shape"
                    );
                    continue;
                }
                quant::coop_matmul(
                    ctx,
                    "bench",
                    act,
                    *w,
                    w.biases.expect("biases"),
                    packed,
                    (ROWS as i32, (2 * intermediate) as i32, k as i32, (2 * intermediate) as i32, k as i32),
                    1,
                    None,
                )
                .expect("matmul");
                kernels_metal::linear::mlp::swiglu(ctx, packed, intermediate, *h).expect("swiglu");
            }
        };
        let us = b.time(&run) / f64::from(layers);
        let out: Vec<Vec<u8>> = hs.iter().map(|h| b.read(*h)).collect();
        match &reference {
            None => reference = Some(out),
            Some(r) => assert!(*r == out, "the fused gate/up differs from the two launches"),
        }
        line += &format!(
            "  {} {us:7.1}us {:5.0}GB/s",
            if fused { "one" } else { "two" },
            bytes as f64 / us / 1e3
        );
    }
    eprintln!("{line}");
}

// EXPERIMENT: the tile's loads alone, its matmuls alone, longer runs of loads, wider threadgroups.
#[test]
#[ignore = "a microbenchmark"]
fn the_coop_variants_price_their_matmuls() {
    let Ok(device) = Context::bind() else { return };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let b = Bench {
        device: &device,
        handles: &handles,
        pipelines: &pipelines,
        keep: RefCell::new(Vec::new()),
    };
    let divisor: u32 = std::env::var("PIE_COOP_BENCH_DIVISOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let plane = Buffer::zeroed(&device, 64 << 20).expect("partials");
    let partials = Tensor::new(
        handles.bind(&plane, 0, plane.bytes()).expect("bind"),
        1,
        16 << 20,
        Dtype::F32,
    );
    for (what, k, n, split) in [
        ("down", 17408u32, 5120u32, 4i32),
        ("gdn/attn out", 6144, 5120, 4),
        ("gate_up", 5120, 34816, 1),
        ("gdn in_proj", 5120, 16384, 1),
    ] {
        let n = n / divisor;
        let bytes = u64::from(k) * u64::from(n) * 9 / 16;
        let layers = STREAM.div_ceil(bytes).clamp(2, 64) as u32;
        let banks: Vec<Bank> = (0..layers)
            .map(|l| b.bank(u64::from(l) * 7919 + u64::from(n), k, n))
            .collect();
        let x: Vec<u8> = fill(u64::from(k) ^ 0xabcd, u64::from(ROWS) * u64::from(k) * 2)
            .chunks(2)
            .flat_map(|c| bf16((f32::from(c[0]) - 127.5) / 256.0))
            .collect();
        let act = Tensor::new(b.upload(&x), ROWS, k, Dtype::Bf16);
        let ys: Vec<Tensor> = (0..layers)
            .map(|_| Tensor::new(b.zeroed(u64::from(ROWS * n) * 2), ROWS, n, Dtype::Bf16))
            .collect();
        let shape = (ROWS as i32, n as i32, k as i32, n as i32, k as i32);
        let mut line = format!("{what:13} K={k:5} N={n:6} x{layers:2} k{split}");
        let base = b.time(&|ctx| {
            for (w, y) in banks.iter().zip(&ys) {
                quant::coop_matmul(ctx, "bench", act, *w, w.biases.expect("b"), *y, shape, split, (split > 1).then_some(partials))
                    .expect("matmul");
            }
        }) / f64::from(layers);
        let want = ys.iter().fold(0u64, |h, y| h.rotate_left(5) ^ hash(&b.read(*y)));
        line += &format!("  base {base:6.1}us {:4.0}GB/s", bytes as f64 / base / 1e3);
        for variant in [(0, 1, 2), (1, 1, 2), (3, 1, 2), (2, 1, 2), (0, 1, 4), (0, 1, 1)] {
            let us = b.time(&|ctx| {
                for (w, y) in banks.iter().zip(&ys) {
                    quant::coop_matmul_x(ctx, act, *w, *y, shape, split, (split > 1).then_some(partials), variant)
                        .expect("variant");
                }
            }) / f64::from(layers);
            let same = variant.0 != 0
                || ys.iter().fold(0u64, |h, y| h.rotate_left(5) ^ hash(&b.read(*y))) == want;
            line += &format!(
                "  m{}r{}s{} {us:6.1}us {:4.0}GB/s{}",
                variant.0,
                variant.1,
                variant.2,
                bytes as f64 / us / 1e3,
                if same { "" } else { "!" }
            );
        }
        eprintln!("{line}");
        b.keep.borrow_mut().clear();
    }
}
