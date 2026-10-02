#![cfg(target_vendor = "apple")]

//! Step-1 measurement bench (NOT a correctness gate, though it checks row-0
//! invariance): time the CURRENT ternary qmv kernels (`pq2_0_qmv`, `ptq1_0_qmv`)
//! across a prefill row ladder on model-representative shapes, and the affine
//! `quant::matmul` dispatch (qmv -> qmm_t) at the same shapes as the "what tiling
//! could buy" ceiling.
//!
//! The ternary qmv grid is `qmv_grid`: one threadgroup per (activation vector,
//! 8 output rows). For `m` activation vectors the weight matrix is read `m` times
//! with NO reuse, so us/row is expected flat (bandwidth-bound). The affine path
//! routes to the simdgroup-fragment `qmm_t` above a batch threshold and reuses
//! each weight tile across BM rows, so its us/row is expected to collapse as `m`
//! grows. The gap between the two us/row curves is the recoverable headroom.
//!
//! Env knobs: PIE_TERN_SHAPES ("KxN,KxN"), PIE_TERN_ROWS ("1,128,512,2048"),
//! PIE_TERN_STEPS, PIE_TERN_WARM_MS. Default shapes/rows match the task brief.

use std::time::Instant;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

const PTQ1_0_FILE: &str = "linear/quant_ptq1_0.metal";
const PQ2_0_FILE: &str = "linear/quant_pq2_0.metal";
const QMV_GROUP: [u32; 3] = [32, 2, 1];

const PTQ1_0_BYTES: u32 = 28;
const PQ2_0_BYTES: u32 = 34;
const BLOCK: u32 = 128;

const AFFINE_GROUP: u32 = 64;
const AFFINE_BITS: u32 = 4;

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
    std::env::var("PIE_TERN_SHAPES")
        .unwrap_or_else(|_| "5120x17408,17408x5120,5120x6144,2048x5120".to_string())
        .split(',')
        .filter_map(|s| {
            let (k, n) = s.trim().split_once('x')?;
            Some((k.trim().parse().ok()?, n.trim().parse().ok()?))
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Codec {
    Ptq1_0,
    Pq2_0,
}

impl Codec {
    fn file(self) -> &'static str {
        match self {
            Codec::Ptq1_0 => PTQ1_0_FILE,
            Codec::Pq2_0 => PQ2_0_FILE,
        }
    }
    fn entry(self) -> &'static str {
        match self {
            Codec::Ptq1_0 => "ptq1_0_qmv_bfloat16",
            Codec::Pq2_0 => "pq2_0_qmv_bfloat16",
        }
    }
    fn block_bytes(self) -> u32 {
        match self {
            Codec::Ptq1_0 => PTQ1_0_BYTES,
            Codec::Pq2_0 => PQ2_0_BYTES,
        }
    }
    fn dtype(self) -> Dtype {
        match self {
            Codec::Ptq1_0 => Dtype::Ptq1_0,
            Codec::Pq2_0 => Dtype::Pq2_0,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Codec::Ptq1_0 => "PTQ1_0 (1.75bpw)",
            Codec::Pq2_0 => "PQ2_0 (2.125bpw)",
        }
    }
    /// Lay one row of `num_blocks` blocks: a sane fp16 scale per block (so the
    /// decoded weights and the dot are finite) plus random code bytes. We are
    /// timing, not validating a codec, so any byte pattern decodes to a valid
    /// trit/code and the per-row result only has to be finite and stable.
    fn write_block(self, blk: &mut [u8], seed: u64) {
        let scale = bf16(0.012 + 0.001 * f32::from(noise(seed) % 8));
        for (at, byte) in blk.iter_mut().enumerate() {
            *byte = noise(seed ^ (at as u64).wrapping_mul(0x100));
        }
        match self {
            // Trailing fp16 scale at bytes 26-27.
            Codec::Ptq1_0 => blk[26..28].copy_from_slice(&scale),
            // Leading fp16 scale at bytes 0-1.
            Codec::Pq2_0 => blk[0..2].copy_from_slice(&scale),
        }
    }
}

#[test]
fn the_ternary_prefill_is_priced_by_its_rows_every_case() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    let rows = list("PIE_TERN_ROWS", "1,128,512,2048");
    let steps: usize = env("PIE_TERN_STEPS", 30usize);
    let warm_ms: u64 = env("PIE_TERN_WARM_MS", 250u64);

    for (k, n) in shapes() {
        assert!(
            k.is_multiple_of(BLOCK),
            "K={k} must be a whole number of 128-blocks"
        );
        eprintln!("\n================  K={k}  N={n}  ================");
        for codec in [Codec::Ptq1_0, Codec::Pq2_0] {
            time_qmv(
                &device, &handles, &pipelines, codec, k, n, &rows, steps, warm_ms,
            );
        }
        time_pq2_0_tiled(&device, &handles, &pipelines, k, n, &rows, steps, warm_ms);
        time_affine_ceiling(&device, &handles, &pipelines, k, n, &rows, steps, warm_ms);
    }
}

#[allow(clippy::too_many_arguments)]
fn time_qmv(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    codec: Codec,
    k: u32,
    n: u32,
    rows: &[u32],
    steps: usize,
    warm_ms: u64,
) {
    let num_blocks = k / BLOCK;
    let row_bytes = num_blocks * codec.block_bytes();
    let codes_bytes = u64::from(n) * u64::from(row_bytes);
    let widest = *rows.iter().max().expect("a row");

    let mut codes_b = Buffer::zeroed(device, codes_bytes).expect("codes");
    {
        let mut codes = vec![0u8; usize::try_from(codes_bytes).expect("codes fit")];
        for r in 0..n {
            for blk in 0..num_blocks {
                let at = (u64::from(r) * u64::from(num_blocks) + u64::from(blk))
                    * u64::from(codec.block_bytes());
                let at = usize::try_from(at).expect("offset fits");
                let end = at + codec.block_bytes() as usize;
                codec.write_block(&mut codes[at..end], u64::from(r) ^ (u64::from(blk) << 20));
            }
        }
        codes_b.write(0, &codes).expect("write codes");
    }
    let mut act_b = Buffer::zeroed(device, u64::from(widest) * u64::from(k) * 2).expect("act");
    {
        let mut act =
            vec![0u8; usize::try_from(u64::from(widest) * u64::from(k) * 2).expect("act fits")];
        for (at, pair) in act.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(0.02 * (f32::from(noise(at as u64) % 16) - 8.0)));
        }
        act_b.write(0, &act).expect("write act");
    }
    let out_b = Buffer::zeroed(device, u64::from(widest) * u64::from(n) * 2).expect("out");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hc, ha, ho) = (bind(&codes_b), bind(&act_b), bind(&out_b));

    let (ki, ni) = (
        i32::try_from(k).expect("k fits"),
        i32::try_from(n).expect("n fits"),
    );

    eprintln!(
        "  {} qmv  ({:.1} MiB codes, {} B/row)",
        codec.label(),
        codes_bytes as f64 / (1u64 << 20) as f64,
        row_bytes,
    );

    let mut one_per_row = 0.0f64;
    let mut reference: Option<Vec<f32>> = None;
    for &m in rows {
        let batch = (1024u32 / m).max(1) as usize;
        let launch = |sink: &dyn Encode| {
            sink.fire(
                Fire::at(codec.file(), codec.entry()).apply(Grid::of(
                    quant::qmv_grid("bench", i32::try_from(m).expect("m"), ni).expect("grid"),
                    QMV_GROUP,
                )),
                &[
                    Tensor::new(hc, n, k, codec.dtype()).arg(),
                    Tensor::new(ha, m, k, Dtype::Bf16).arg(),
                    Tensor::new(ho, m, n, Dtype::Bf16).arg_mut(),
                    ki.arg(),
                    ni.arg(),
                ],
            )
            .expect("the qmv launch");
        };

        // Warm.
        let began = Instant::now();
        loop {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(device, &frame, pipelines, handles);
            for _ in 0..batch {
                launch(&sink);
            }
            frame.commit().expect("warm commit");
            if began.elapsed().as_millis() as u64 >= warm_ms {
                break;
            }
        }

        // Timed.
        let mut device_s = 0.0f64;
        for _ in 0..steps {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(device, &frame, pipelines, handles);
            for _ in 0..batch {
                launch(&sink);
            }
            device_s += frame.commit_timed().expect("the commit");
        }
        let launches = (steps * batch) as f64;

        // Row-0 invariance: qmv computes each output row independently of m, so
        // row 0 must read back identically for every m (a real defect breaks it).
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
                for (a, b) in want.iter().zip(&got) {
                    let scale = f64::from(a.abs()).max(f64::from(b.abs())).max(1e-3);
                    worst = worst.max(f64::from((a - b).abs()) / scale);
                }
                assert!(
                    worst <= 1e-6,
                    "rows {m} answers row 0 differently from rows {}: worst relative {worst:.2e}",
                    rows[0]
                );
            }
        }

        let dev_us = device_s * 1e6 / launches;
        let per_row = dev_us / f64::from(m);
        if m == rows[0] {
            one_per_row = per_row;
        }
        // "single-matrix" GB/s mirrors the affine bench's convention (codes
        // bytes / time); the weight-traffic GB/s is the real bus pressure, which
        // for qmv is codes * m (every row re-reads the whole matrix).
        let single_gb = codes_bytes as f64 / 1e9;
        let traffic_gbps = single_gb * f64::from(m) / (dev_us / 1e6);
        eprintln!(
            "    rows {m:>4}: {dev_us:>9.1} us  {per_row:>7.2} us/row  \
             ({:>5.2}x the m=1 us/row)  {traffic_gbps:>6.1} GB/s weight traffic",
            per_row / one_per_row.max(1e-12),
        );
    }
}

/// The GEMM-tiled PQ2_0 qmm via the `quant::matmul` dispatch (routes to the
/// tiled kernel for m >= the threshold), at the same shapes. Compare its us/row
/// to the PQ2_0 qmv above: that ratio is the prefill speedup the tiling buys.
#[allow(clippy::too_many_arguments)]
fn time_pq2_0_tiled(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    k: u32,
    n: u32,
    rows: &[u32],
    steps: usize,
    warm_ms: u64,
) {
    let num_blocks = k / BLOCK;
    let row_bytes = num_blocks * PQ2_0_BYTES;
    let codes_bytes = u64::from(n) * u64::from(row_bytes);
    let widest = *rows.iter().max().expect("a row");
    // Capacity holds the padded rows; the dispatch pads m up to its row tile.
    let cap = widest.div_ceil(64).max(1) * 64;

    let mut codes_b = Buffer::zeroed(device, codes_bytes).expect("codes");
    {
        let mut codes = vec![0u8; usize::try_from(codes_bytes).expect("codes fit")];
        for r in 0..n {
            for blk in 0..num_blocks {
                let at = (u64::from(r) * u64::from(num_blocks) + u64::from(blk))
                    * u64::from(PQ2_0_BYTES);
                let at = usize::try_from(at).expect("offset fits");
                Codec::Pq2_0.write_block(
                    &mut codes[at..at + PQ2_0_BYTES as usize],
                    u64::from(r) ^ (u64::from(blk) << 20),
                );
            }
        }
        codes_b.write(0, &codes).expect("write codes");
    }
    let mut act_b = Buffer::zeroed(device, u64::from(cap) * u64::from(k) * 2).expect("act");
    {
        let mut act =
            vec![0u8; usize::try_from(u64::from(cap) * u64::from(k) * 2).expect("act fits")];
        for (at, pair) in act.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(0.02 * (f32::from(noise(at as u64) % 16) - 8.0)));
        }
        act_b.write(0, &act).expect("write act");
    }
    let out_b = Buffer::zeroed(device, u64::from(cap) * u64::from(n) * 2).expect("out");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hc, ha, ho) = (bind(&codes_b), bind(&act_b), bind(&out_b));

    eprintln!(
        "  PQ2_0 tiled qmm (dispatch)  ({:.1} MiB codes)",
        codes_bytes as f64 / (1u64 << 20) as f64,
    );

    let mut one_per_row = 0.0f64;
    for &m in rows {
        let batch = (1024u32 / m).max(1) as usize;
        let bank = Bank {
            codes: Tensor::new(hc, n, k, Dtype::Pq2_0),
            scales: Tensor::new(hc, n, 1, Dtype::Bf16),
            biases: None,
            group: 128,
            bits: 2,
        };
        let act = Tensor::new(ha, m, k, Dtype::Bf16);
        let y = Tensor::new(ho, m, n, Dtype::Bf16);
        let none = |_: u32, _: u32| None;
        let launch = |sink: &dyn Encode| {
            let scratch = quant::Scratch {
                precast: &none,
                partials: &none,
            };
            quant::matmul(sink, act, bank, y, scratch, cap).expect("the tiled launch");
        };

        let began = Instant::now();
        loop {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(device, &frame, pipelines, handles);
            for _ in 0..batch {
                launch(&sink);
            }
            frame.commit().expect("warm commit");
            if began.elapsed().as_millis() as u64 >= warm_ms {
                break;
            }
        }

        let mut device_s = 0.0f64;
        for _ in 0..steps {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(device, &frame, pipelines, handles);
            for _ in 0..batch {
                launch(&sink);
            }
            device_s += frame.commit_timed().expect("the commit");
        }
        let launches = (steps * batch) as f64;
        let dev_us = device_s * 1e6 / launches;
        let per_row = dev_us / f64::from(m);
        if m == rows[0] {
            one_per_row = per_row;
        }
        eprintln!(
            "    rows {m:>4}: {dev_us:>9.1} us  {per_row:>7.2} us/row  \
             ({:>5.2}x the m=1 us/row)",
            per_row / one_per_row.max(1e-12),
        );
    }
}

/// The affine `quant::matmul` dispatch (U4g64, b=4) at the same shapes — the
/// production path that routes qmv -> simdgroup-fragment qmm_t above a batch
/// threshold. Its us/row curve is the achievable ceiling a tiled kernel reaches.
#[allow(clippy::too_many_arguments)]
fn time_affine_ceiling(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    k: u32,
    n: u32,
    rows: &[u32],
    steps: usize,
    warm_ms: u64,
) {
    let group = AFFINE_GROUP;
    let bits = AFFINE_BITS;
    let words = u64::from(n) * u64::from(k) * u64::from(bits) / 32;
    let factors_n = u64::from(n) * u64::from(k / group);
    let widest = *rows.iter().max().expect("a row");
    let cap = rows
        .iter()
        .map(|&m| {
            quant::bm_rung(i32::try_from(m).unwrap()).unsigned_abs()
                * m.div_ceil(quant::bm_rung(i32::try_from(m).unwrap()).unsigned_abs())
        })
        .max()
        .unwrap()
        .max(widest);

    let mut codes_b = Buffer::zeroed(device, words * 4).expect("codes");
    let mut scales_b = Buffer::zeroed(device, factors_n * 2).expect("scales");
    let mut biases_b = Buffer::zeroed(device, factors_n * 2).expect("biases");
    let mut act_b = Buffer::zeroed(device, u64::from(cap) * u64::from(k) * 2).expect("act");
    let out_b = Buffer::zeroed(device, u64::from(cap) * u64::from(n) * 2).expect("out");
    let precast_b = Buffer::zeroed(device, u64::from(cap) * u64::from(k) * 2).expect("precast");
    let partial_b =
        Buffer::zeroed(device, u64::from(8u32.max(cap)) * u64::from(n) * 4 * 8).expect("partials");
    {
        let mut codes = vec![0u8; usize::try_from(words * 4).expect("codes fit")];
        for (at, byte) in codes.iter_mut().enumerate() {
            *byte = noise(at as u64);
        }
        codes_b.write(0, &codes).expect("write codes");
        let mut factors = vec![0u8; usize::try_from(factors_n * 2).expect("factors fit")];
        for (at, pair) in factors.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(0.01 + 0.001 * f32::from(noise(at as u64 ^ 0xAA) % 8)));
        }
        scales_b.write(0, &factors).expect("write scales");
        for (at, pair) in factors.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(-0.05 + 0.01 * f32::from(noise(at as u64 ^ 0x55) % 8)));
        }
        biases_b.write(0, &factors).expect("write biases");
        let mut act =
            vec![0u8; usize::try_from(u64::from(cap) * u64::from(k) * 2).expect("act fits")];
        for (at, pair) in act.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(0.02 * (f32::from(noise(at as u64) % 16) - 8.0)));
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
    let partial_rows = partial_b.bytes() / (u64::from(n) * 4);

    eprintln!(
        "  AFFINE U4g64 b4 qmm ceiling  ({:.1} MiB codes)",
        words as f64 * 4.0 / (1u64 << 20) as f64,
    );

    let mut one_per_row = 0.0f64;
    for &m in rows {
        let batch = (1024u32 / m).max(1) as usize;
        let bank = Bank {
            codes: Tensor::new(hc, n, k, Dtype::U4g64),
            scales: Tensor::new(hs, n, k / group, Dtype::Bf16),
            biases: Some(Tensor::new(hb, n, k / group, Dtype::Bf16)),
            group,
            bits,
        };
        let act = Tensor::new(ha, m, k, Dtype::Bf16);
        let y = Tensor::new(ho, m, n, Dtype::Bf16);
        let some = |rows: u32, contraction: u32| {
            (u64::from(rows) * u64::from(contraction) <= u64::from(cap) * u64::from(k))
                .then(|| Tensor::new(hp, rows, contraction, Dtype::F16))
        };
        let some_partials = |rows: u32, width: u32| {
            (width == n && u64::from(rows) <= partial_rows)
                .then(|| Tensor::new(hq, rows, width, Dtype::F32))
        };
        let precast: &dyn Fn(u32, u32) -> Option<Tensor> = &some;
        let partials: &dyn Fn(u32, u32) -> Option<Tensor> = &some_partials;
        let launch = |sink: &dyn Encode| {
            let scratch = quant::Scratch { precast, partials };
            quant::matmul(sink, act, bank, y, scratch, cap).expect("the affine launch");
        };

        let began = Instant::now();
        loop {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(device, &frame, pipelines, handles);
            for _ in 0..batch {
                launch(&sink);
            }
            frame.commit().expect("warm commit");
            if began.elapsed().as_millis() as u64 >= warm_ms {
                break;
            }
        }

        let mut device_s = 0.0f64;
        for _ in 0..steps {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(device, &frame, pipelines, handles);
            for _ in 0..batch {
                launch(&sink);
            }
            device_s += frame.commit_timed().expect("the commit");
        }
        let launches = (steps * batch) as f64;
        let dev_us = device_s * 1e6 / launches;
        let per_row = dev_us / f64::from(m);
        if m == rows[0] {
            one_per_row = per_row;
        }
        let single_gb = words as f64 * 4.0 / 1e9;
        eprintln!(
            "    rows {m:>4}: {dev_us:>9.1} us  {per_row:>7.2} us/row  \
             ({:>5.2}x the m=1 us/row)  {:>6.1} GB/s codes",
            per_row / one_per_row.max(1e-12),
            single_gb / (dev_us / 1e6),
        );
    }
}
