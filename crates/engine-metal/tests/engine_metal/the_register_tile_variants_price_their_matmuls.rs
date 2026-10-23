#![cfg(target_vendor = "apple")]
//! Microbench (not a test): the Apple10 register-fed (cooperative) tile over the dense
//! projections of a Qwen3.8-27B speculative verify (8 rows, u4g64 affine), each projection's
//! layers in distinct banks so the weights stream from memory as a fire's do. Per split, the
//! device time of one launch (with its reduction) and the weight bytes it streams a second, and
//! a hash of the outputs, so a change to the tile can be held to the same bits.
//! `PIE_COOP_BENCH_DIVISOR` divides every projection's columns, to stand a GPU of fewer cores
//! in for a wider one (4 on a 10-core M5 for a 40-core M5 Max).
use std::cell::RefCell;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Fire, Grid};
use kernels_metal::linear::quant;
use kernels_metal::{Arg, Bank, Encode, Tensor};
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

fn sym(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[test]
#[ignore = "a microbenchmark"]
fn the_register_tile_variants_price_their_matmuls() {
    let Ok(device) = Context::bind() else { return };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let b = Bench {
        device: &device,
        handles: &handles,
        pipelines: &pipelines,
        keep: RefCell::new(Vec::new()),
    };
    let cores = device.cores();
    eprintln!("cores {cores:?}");
    // (U, NF, SIMDS, VS); None = the shipped kernel
    let variants: [Option<(u32, u32, u32, u32)>; 15] = [
        None,
        Some((2, 2, 4, 0)),
        Some((4, 2, 4, 0)),
        Some((2, 2, 4, 1)),
        Some((4, 2, 4, 1)),
        Some((1, 2, 4, 0)),
        Some((2, 4, 2, 0)),
        Some((2, 4, 4, 0)),
        Some((4, 4, 2, 1)),
        Some((2, 2, 2, 0)),
        Some((2, 2, 8, 0)),
        Some((2, 1, 4, 0)),
        Some((4, 1, 8, 1)),
        Some((2, 2, 4, 2)),
        Some((2, 2, 4, 3)),
    ];
    let mut totals = vec![0.0f64; variants.len()];
    // (what, K, N, layers a verify holds): the 27B verify's register projections
    for (what, k, n, per) in [
        ("gate_up", 5120u32, 34816u32, 64u32),
        ("down", 17408, 5120, 64),
        ("gdn in_proj", 5120, 16384, 48),
        ("out", 6144, 5120, 64),
        ("attn q+gate", 5120, 12288, 16),
        ("head", 5120, 248320, 1),
    ] {
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
        let split = quant::register_split(n as i32, k as i32, cores);
        let counters = if split > 1 { n / 64 } else { 0 };
        let table_rows = (16 * k + k + 4 * counters).div_ceil(2 * k);
        let table = Tensor::new(b.zeroed(u64::from(table_rows * k) * 2), table_rows, k, Dtype::F16);
        let partial = Tensor::new(
            b.zeroed(u64::from(ROWS) * u64::from(split as u32) * u64::from(n) * 4),
            ROWS * split as u32,
            n,
            Dtype::F32,
        );
        {
            let frame = device.frame().expect("frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            quant::register_table(&sink, act, table, ROWS as i32, k as i32, counters as i32)
                .expect("table");
            frame.commit().expect("commit");
        }
        let ys: Vec<Tensor> = (0..layers)
            .map(|_| Tensor::new(b.zeroed(u64::from(ROWS * n) * 2), ROWS, n, Dtype::Bf16))
            .collect();
        let mut line = format!("{what:12} K={k:5} N={n:6} k{split} x{layers:2}");
        let mut want = None;
        for (vi, v) in variants.iter().enumerate() {
            let (entry, stamp, cols_tg, threads) = match *v {
                None => {
                    let e = sym(format!("affine_qmm_sg_k{split}"));
                    (e, sym(format!("PIE_SG_POINT(\"{e}\",{split})")), 64, 128)
                }
                Some((u, nf, simds, vs)) => {
                    let e = sym(format!("affine_qmm_sgx_k{split}_u{u}_f{nf}_s{simds}_v{vs}"));
                    (
                        e,
                        sym(format!("PIE_SGX(\"{e}\",{split},{u},{nf},{simds},{vs})")),
                        8 * nf * simds,
                        32 * simds,
                    )
                }
            };
            if n % cols_tg != 0 {
                line += "  -";
                continue;
            }
            let run = |ctx: &Sink<'_>| {
                for (w, y) in banks.iter().zip(&ys) {
                    ctx.fire(
                        Fire::at("linear/quant_qmm_sg.metal", entry).stamp(stamp).apply(Grid::of(
                            [threads * (n / cols_tg), split as u32, 1],
                            [threads, 1, 1],
                        )),
                        &[
                            w.codes.arg(),
                            w.scales.arg(),
                            w.biases.expect("biases").arg(),
                            table.arg(),
                            y.arg_mut(),
                            (k as i32).arg(),
                            (n as i32).arg(),
                            if split > 1 { partial.arg_mut() } else { ctx.absent().expect("absent") },
                            (ROWS as i32).arg(),
                            (k as i32).arg(),
                        ],
                    )
                    .expect("fire");
                }
            };
            let us = b.time(&run) / f64::from(layers);
            let h = ys
                .iter()
                .fold(0u64, |h, y| h.rotate_left(5) ^ hash(&b.read(*y)));
            let same = match want {
                None => {
                    want = Some(h);
                    "="
                }
                Some(w) if w == h => "=",
                Some(_) => "!",
            };
            totals[vi] += us * f64::from(per) / 1e3;
            line += &format!("  v{vi} {us:7.1}us {:4.0}GB/s{same}", bytes as f64 / us / 1e3);
        }
        eprintln!("{line}");
        b.keep.borrow_mut().clear();
    }
    for (vi, v) in variants.iter().enumerate() {
        eprintln!("v{vi} {v:?}: verify's register matmuls {:.2} ms", totals[vi]);
    }
}
