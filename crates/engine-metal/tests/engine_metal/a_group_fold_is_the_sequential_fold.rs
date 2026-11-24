#![cfg(target_vendor = "apple")]
//! A beam group's folds (BEAM-PAGE-LISTS §11.3) against the sequential
//! recurrence, over Qwen3.6-35B-A3B's heads: `n` buffered rows (raw keys,
//! corrected values `u`, `[b | a]`, the conv's input rows) read through a
//! lane's page table row started `n` rows past them, as a beam lane's is.
//! The gated-delta fold leaves each value head's state as
//! `S <- g S + k u^T` a row in turn (each key normed, each gate made from
//! its `a`) does, held in f64; the conv fold leaves the window's last rows.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::ssm::{self, Corrected, GroupFolds, Paged};
use kernels_metal::tensor::RecurrentPool;
use kernels_metal::{StateMap, Tensor};
use poem_ir::Dtype;

const K_HEADS: u32 = 16;
const V_HEADS: u32 = 32;
const DIM: u32 = 128;
const BANK_FLOATS: u64 = V_HEADS as u64 * DIM as u64 * DIM as u64;
/// Tokens a page of the test's buffer, so the folded rows cross pages.
const PAGE: u32 = 4;
const PAGE_INTS: u32 = 11;
/// The in-page token of the lane's first buffered row.
const FIRST: u32 = 2;
const CHANNELS: u32 = 64;
const TAPS: u32 = 4;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

struct Bench {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    keep: std::cell::RefCell<Vec<Buffer>>,
}

impl Bench {
    fn new() -> Option<Bench> {
        let device = Context::bind().ok()?;
        Some(Bench {
            device,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
            keep: std::cell::RefCell::new(Vec::new()),
        })
    }

    fn upload(&self, bytes: &[u8], rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let mut buffer =
            Buffer::zeroed(&self.device, (bytes.len() as u64).max(16)).expect("a buffer");
        buffer.write(0, bytes).expect("an upload");
        let handle = self
            .handles
            .bind(&buffer, 0, buffer.bytes())
            .expect("a handle");
        self.keep.borrow_mut().push(buffer);
        Tensor::new(handle, rows, width, dtype)
    }

    fn i32s(&self, v: &[i32]) -> Tensor {
        self.upload(&ints(v), v.len() as u32, 1, Dtype::I32)
    }

    fn map(&self, stride: u64) -> Tensor {
        let map = StateMap::whole(stride);
        self.upload(&map.bytes(), 1, map.words().len() as u32, Dtype::U64)
    }

    fn floats(&self, t: Tensor, n: usize) -> Vec<f32> {
        let bytes = self.handles.read(t.buf, n as u64 * 4).expect("a read back");
        bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    }

    fn run(&self, fire: &dyn Fn(&Sink)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        fire(&sink);
        frame.commit().expect("the commit");
    }
}

/// `rows` rows of `row_bytes` laid in a paged slab as the lane's buffered
/// rows from in-page token [`FIRST`], its pages in descending slots; the
/// slab, its page bytes, and the lane's page table row started `skip` rows
/// on.
fn paged(bench: &Bench, rows: &[Vec<u8>], row_bytes: u32, skip: u32) -> (Tensor, u32, Vec<i32>) {
    let pages = (PAGE_INTS - 1) as usize;
    let page_bytes = (u64::from(PAGE) * u64::from(row_bytes)).next_multiple_of(256) as usize;
    let mut slab = vec![0u8; pages * page_bytes];
    let slot = |page: usize| pages - 1 - page;
    for (e, row) in rows.iter().enumerate() {
        let token = FIRST as usize + e;
        let at = slot(token / PAGE as usize) * page_bytes + (token % PAGE as usize) * row_bytes as usize;
        slab[at..at + row.len()].copy_from_slice(row);
    }
    let mut table = vec![(FIRST + skip) as i32];
    table.extend((0..pages).map(|page| slot(page) as i32));
    (
        bench.upload(&slab, pages as u32, page_bytes as u32, Dtype::U8),
        page_bytes as u32,
        table,
    )
}

fn softplus(z: f64) -> f64 {
    if z > 20.0 { z } else { (1.0 + z.exp()).ln() }
}

#[test]
fn a_group_fold_is_the_sequential_fold() {
    let Some(bench) = Bench::new() else {
        eprintln!("no Metal device; skipped");
        return;
    };
    for n in [1u32, 5, 13, 30] {
        gated_delta(&bench, n);
        conv(&bench, n);
    }
}

fn gated_delta(bench: &Bench, n: u32) {
    let keys_w = (K_HEADS * DIM) as usize;
    let u_w = (V_HEADS * DIM) as usize;
    let ba_w = (2 * V_HEADS) as usize;
    let salt = u64::from(n) << 40;
    let keys: Vec<Vec<f32>> = (0..n as u64)
        .map(|t| (0..keys_w as u64).map(|i| bf16_round(unit(salt ^ (t << 20) ^ i ^ 0x1111))).collect())
        .collect();
    let us: Vec<Vec<f32>> = (0..n as u64)
        .map(|t| (0..u_w as u64).map(|i| 0.5 * unit(salt ^ (t << 20) ^ i ^ 0x2222)).collect())
        .collect();
    let bas: Vec<Vec<f32>> = (0..n as u64)
        .map(|t| (0..ba_w as u64).map(|i| bf16_round(3.0 * unit(salt ^ (t << 20) ^ i ^ 0x3333))).collect())
        .collect();
    let dt_bias: Vec<f32> = (0..V_HEADS as u64).map(|i| bf16_round(unit(i ^ 0x4444))).collect();
    let a_log: Vec<f32> = (0..V_HEADS as u64).map(|i| unit(i ^ 0x5555)).collect();
    let bank: Vec<f32> = (0..BANK_FLOATS).map(|at| 0.1 * unit(at ^ 0xC3C3)).collect();

    // One page size for every plane, as one buffer page holds them all.
    let page_bytes = (u64::from(PAGE) * u_w as u64 * 4).next_multiple_of(256) as u32;
    let relaid = |rows: &[Vec<u8>], row_bytes: u32| -> Tensor {
        let pages = (PAGE_INTS - 1) as usize;
        let mut slab = vec![0u8; pages * page_bytes as usize];
        for (e, row) in rows.iter().enumerate() {
            let token = FIRST as usize + e;
            let at = (pages - 1 - token / PAGE as usize) * page_bytes as usize
                + (token % PAGE as usize) * row_bytes as usize;
            slab[at..at + row.len()].copy_from_slice(row);
        }
        bench.upload(&slab, pages as u32, page_bytes, Dtype::U8)
    };
    let key_slab = relaid(&keys.iter().map(|k| bf16_bytes(k)).collect::<Vec<_>>(), keys_w as u32 * 2);
    let ba_slab = relaid(&bas.iter().map(|b| bf16_bytes(b)).collect::<Vec<_>>(), ba_w as u32 * 2);
    let u_slab = relaid(&us.iter().map(|u| f32_bytes(u)).collect::<Vec<_>>(), u_w as u32 * 4);
    let pages_a_lane = (PAGE_INTS - 1) as i32;
    let mut table = vec![(FIRST + n) as i32];
    table.extend((0..pages_a_lane).map(|page| pages_a_lane - 1 - page));

    let state = bench.upload(&f32_bytes(&bank), V_HEADS * DIM, DIM, Dtype::F32);
    let pool = RecurrentPool {
        state,
        slots: bench.i32s(&[0]),
        conv_state: state,
        new_conv_state: state,
        segments: bench.map(BANK_FLOATS),
    };
    let pages = bench.i32s(&table);
    let folds = GroupFolds {
        table: bench.i32s(&[0, n as i32, 0, 0]),
        count: 1,
    };
    bench.run(&|sink| {
        ssm::gated_delta_group_fold(
            sink,
            folds,
            (0, 1),
            bench.i32s(&[0]),
            (
                bench.upload(&f32_bytes(&a_log), V_HEADS, 1, Dtype::F32),
                bench.upload(&bf16_bytes(&dt_bias), V_HEADS, 1, Dtype::Bf16),
            ),
            Paged {
                slab: ba_slab,
                pages,
                page_tokens: PAGE,
                page_bytes,
                row_bytes: ba_w as u32 * 2,
                page_ints: PAGE_INTS,
            },
            Corrected {
                keys: key_slab,
                key_row_bytes: keys_w as u32 * 2,
                u: u_slab,
                u_row_bytes: u_w as u32 * 4,
                // The folds read no scalars.
                scalars: u_slab,
                scalars_row_bytes: u_w as u32 * 4,
            },
            &pool,
            (K_HEADS, V_HEADS, DIM, DIM),
        )
        .expect("the gated-delta fold encodes");
    });
    let got = bench.floats(state, BANK_FLOATS as usize);

    // The sequential recurrence in f64.
    let d = DIM as usize;
    let mut want: Vec<f64> = bank.iter().map(|&x| f64::from(x)).collect();
    for t in 0..n as usize {
        for hv in 0..V_HEADS as usize {
            let hk = hv / (V_HEADS / K_HEADS) as usize;
            let k = &keys[t][hk * d..(hk + 1) * d];
            let kinv = 1.0 / (k.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() + 1e-6).sqrt();
            let z = f64::from(bas[t][V_HEADS as usize + hv]) + f64::from(dt_bias[hv]);
            let g = (-f64::from(a_log[hv]).exp() * softplus(z)).exp();
            for dv in 0..d {
                let u = f64::from(us[t][hv * d + dv]);
                let row = &mut want[(hv * d + dv) * d..(hv * d + dv + 1) * d];
                for (dk, cell) in row.iter_mut().enumerate() {
                    *cell = g * *cell + f64::from(k[dk]) * kinv * u;
                }
            }
        }
    }
    let scale = want.iter().fold(0f64, |m, x| m.max(x.abs()));
    let worst = got
        .iter()
        .zip(&want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0f64, f64::max);
    assert!(
        worst <= 1e-5 * scale.max(1.0),
        "the group fold of {n} rows is {worst:e} from the sequential recurrence (scale {scale:e})"
    );
}

fn conv(bench: &Bench, n: u32) {
    let c = CHANNELS as usize;
    let hist = TAPS as usize;
    let salt = u64::from(n) << 40;
    let xs: Vec<Vec<f32>> = (0..n as u64)
        .map(|t| (0..c as u64).map(|i| bf16_round(unit(salt ^ (t << 20) ^ i ^ 0x7777))).collect())
        .collect();
    let window: Vec<f32> = (0..(hist * c) as u64).map(|at| unit(at ^ 0x6666)).collect();
    let (slab, page_bytes, table) =
        paged(bench, &xs.iter().map(|x| bf16_bytes(x)).collect::<Vec<_>>(), CHANNELS * 2, n);
    let state = bench.upload(&f32_bytes(&window), TAPS, CHANNELS, Dtype::F32);
    let pool = RecurrentPool {
        state,
        slots: bench.i32s(&[0]),
        conv_state: state,
        new_conv_state: state,
        segments: bench.map(u64::from(TAPS * CHANNELS)),
    };
    let folds = GroupFolds {
        table: bench.i32s(&[0, n as i32, 0, 0]),
        count: 1,
    };
    bench.run(&|sink| {
        ssm::causal_conv1d_group_fold(
            sink,
            folds,
            (0, 1),
            bench.i32s(&[0]),
            Paged {
                slab,
                pages: bench.i32s(&table),
                page_tokens: PAGE,
                page_bytes,
                row_bytes: CHANNELS * 2,
                page_ints: PAGE_INTS,
            },
            &pool,
            CHANNELS,
            TAPS,
            1,
        )
        .expect("the conv fold encodes");
    });
    let got = bench.floats(state, hist * c);
    // The window's last `hist` rows of the state's and the folded rows.
    let rows: Vec<&[f32]> = (0..hist)
        .map(|s| &window[s * c..(s + 1) * c])
        .chain(xs.iter().map(Vec::as_slice))
        .collect();
    let want: Vec<f32> = rows[rows.len() - hist..].iter().flat_map(|r| r.iter().copied()).collect();
    assert_eq!(got, want, "the conv fold of {n} rows keeps the window's last {hist} rows");
}
