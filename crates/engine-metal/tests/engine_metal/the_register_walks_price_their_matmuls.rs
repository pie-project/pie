#![cfg(target_vendor = "apple")]
//! Microbench (not a test): the register tile's walks and segment launches over the dense
//! projections of a Qwen3.6-35B-A3B speculative verify and DFlash draft (8 rows, u4g64 affine),
//! each projection's layers in distinct banks so the weights stream from memory as a fire's do,
//! the input table laid beforehand as the producer lays it. Per walk `(U, V)` (groups a step,
//! groups a scale load) and split, the device time of one launch; then each layer's sibling
//! projections (a GDN in-projection and its gates, a query, key and value) as launches of their
//! own and as one segment launch. Every output is checked against the default walk's to the bit.
//! Run: cargo test --release -p engine-metal --test engine_metal \
//!        the_register_walks_price_their_matmuls -- --ignored --nocapture

use std::cell::RefCell;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant::{self, Segment};
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

const FRAMES: usize = 15;
const ROWS: u32 = 8;
const CORES: u32 = 20;

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

/// A layer's table for an 8-row activation of width `k`, laid with `counters` zeroed.
fn table(b: &Bench<'_>, k: u32, counters: u32) -> Tensor {
    let x: Vec<u8> = fill(u64::from(k) ^ 0xabcd, u64::from(ROWS) * u64::from(k) * 2)
        .chunks(2)
        .flat_map(|c| bf16((f32::from(c[0]) - 127.5) / 256.0))
        .collect();
    let act = Tensor::new(b.upload(&x), ROWS, k, Dtype::Bf16);
    let rows = quant::register_table_rows(k as i32, counters as i32);
    let table = Tensor::new(
        b.zeroed(u64::from(rows) * u64::from(k) * 2),
        rows,
        k,
        Dtype::Bf16,
    );
    b.time(&|ctx| {
        quant::register_table(ctx, act, table, ROWS as i32, k as i32, counters as i32)
            .expect("table")
    });
    table
}

#[test]
#[ignore = "a microbenchmark"]
fn the_register_walks_price_their_matmuls() {
    let Ok(device) = Context::bind() else { return };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let b = Bench {
        device: &device,
        handles: &handles,
        pipelines: &pipelines,
        keep: RefCell::new(Vec::new()),
    };
    // the split matmuls' partial sums, kept across the shapes
    let plane = Buffer::zeroed(&device, 64 << 20).expect("partials");
    let partials = Tensor::new(
        handles.bind(&plane, 0, plane.bytes()).expect("bind"),
        1,
        16 << 20,
        Dtype::F32,
    );
    let walks: [(u32, u32); 4] = [quant::REGISTER_WALK, (2, 2), (4, 4), (4, 2)];
    // (what, K, N, layers): the verify's and draft's single projections
    for (what, k, n, layers) in [
        ("gdn/attn out", 4096u32, 2048u32, 40u32),
        ("gdn in_proj", 2048, 12288, 30),
        ("attn qg", 2048, 8192, 10),
        ("lm_head", 2048, 248_320, 1),
        ("shared gate_up", 2048, 1024, 40),
        ("shared down", 512, 2048, 40),
        ("draft down", 6144, 2048, 6),
        ("draft gate_up", 2048, 12288, 6),
        ("draft q", 2048, 4096, 6),
        ("draft o", 4096, 2048, 6),
        ("kv", 2048, 512, 20),
    ] {
        let banks: Vec<Bank> = (0..layers)
            .map(|l| b.bank(u64::from(l) * 7919 + u64::from(n), k, n))
            .collect();
        let t = table(&b, k, n / 64);
        let ys: Vec<Tensor> = (0..layers)
            .map(|_| Tensor::new(b.zeroed(u64::from(ROWS * n) * 2), ROWS, n, Dtype::Bf16))
            .collect();
        let engine = quant::register_split(n as i32, k as i32, Some(CORES));
        let mut line = format!("{what:15} K={k:5} N={n:6} x{layers:2} S={engine}");
        let mut reference: Option<Vec<Vec<u8>>> = None;
        for split in [engine, engine / 2, engine * 2] {
            if split < 1 || split > 8 || (k / 256) < split as u32 {
                continue;
            }
            for &walk in &walks {
                let run = |ctx: &Sink<'_>| {
                    for (w, y) in banks.iter().zip(&ys) {
                        let partial = (split > 1).then_some(partials);
                        quant::register_matmul(
                            ctx,
                            *w,
                            w.biases.expect("biases"),
                            t,
                            *y,
                            (ROWS as i32, n as i32, k as i32, n as i32, k as i32),
                            split,
                            partial,
                            walk,
                        )
                        .expect("matmul");
                    }
                };
                let us = b.time(&run) / f64::from(layers);
                let out: Vec<Vec<u8>> = ys.iter().map(|y| b.read(*y)).collect();
                let same = match (&reference, split == engine) {
                    (None, _) => {
                        reference = Some(out);
                        true
                    }
                    (Some(r), true) => *r == out,
                    (Some(_), false) => true,
                };
                assert!(
                    same,
                    "{what}: walk {walk:?} at split {split} differs from the default walk's bits"
                );
                line += &format!("  s{split} u{}v{} {us:7.1}", walk.0, walk.1);
            }
        }
        eprintln!("{line}");
        b.keep.borrow_mut().clear();
    }
    // (what, K, segment widths, layers): each layer's sibling projections of one table
    for (what, k, widths, layers) in [
        ("gdn in_proj+ba", 2048u32, &[12288u32, 64][..], 30u32),
        ("attn qg+k+v", 2048, &[8192, 512, 512][..], 10),
        ("draft q+k+v", 2048, &[4096, 512, 512][..], 6),
    ] {
        let banks: Vec<Vec<Bank>> = (0..layers)
            .map(|l| {
                widths
                    .iter()
                    .map(|&n| b.bank(u64::from(l) * 104_729 + u64::from(n), k, n))
                    .collect()
            })
            .collect();
        let ys: Vec<Vec<Tensor>> = (0..layers)
            .map(|_| {
                widths
                    .iter()
                    .map(|&n| Tensor::new(b.zeroed(u64::from(ROWS * n) * 2), ROWS, n, Dtype::Bf16))
                    .collect()
            })
            .collect();
        let segments: Vec<Vec<Segment>> = banks
            .iter()
            .zip(&ys)
            .map(|(bs, ys)| {
                bs.iter()
                    .zip(ys)
                    .map(|(w, y)| Segment {
                        w: *w,
                        y: *y,
                        split: quant::register_split(y.width as i32, k as i32, Some(CORES)),
                    })
                    .collect()
            })
            .collect();
        let (counters, _) = quant::segments_scratch(&segments[0], ROWS);
        let t = table(&b, k, counters);
        let mut line = format!("{what:15} K={k:5} x{layers:2}");
        let mut reference: Option<Vec<Vec<u8>>> = None;
        for &walk in &walks[..3] {
            for fused in [false, true] {
                let run = |ctx: &Sink<'_>| {
                    for layer in &segments {
                        if fused {
                            quant::register_segments(
                                ctx,
                                layer,
                                t,
                                ROWS as i32,
                                k as i32,
                                Some(partials),
                                walk,
                            )
                            .expect("segments");
                            continue;
                        }
                        for s in layer {
                            let n = s.y.width as i32;
                            quant::register_matmul(
                                ctx,
                                s.w,
                                s.w.biases.expect("biases"),
                                t,
                                s.y,
                                (ROWS as i32, n, k as i32, n, k as i32),
                                s.split,
                                (s.split > 1).then_some(partials),
                                walk,
                            )
                            .expect("matmul");
                        }
                    }
                };
                let us = b.time(&run) / f64::from(layers);
                let out: Vec<Vec<u8>> = ys.iter().flatten().map(|y| b.read(*y)).collect();
                match &reference {
                    None => reference = Some(out),
                    Some(r) => assert!(
                        *r == out,
                        "{what}: walk {walk:?} fused {fused} differs to the bit"
                    ),
                }
                line += &format!(
                    "  u{}v{} {} {us:7.1}",
                    walk.0,
                    walk.1,
                    if fused { "one" } else { "own" }
                );
            }
        }
        eprintln!("{line}");
        b.keep.borrow_mut().clear();
    }
}
