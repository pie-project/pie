#![cfg(target_vendor = "apple")]

//! The GEMM-tiled PQ2_0 qmm (prefill path) must answer the proven qmv kernel —
//! which the M2b oracle test holds bit-exact against the host decoder — AND the
//! host-decoded f64 reference, across the prefill shape/row grid. A faster but
//! wrong kernel is worthless; correctness is the gate.
//!
//! Three checks per (m, K, N):
//!   1. `quant::matmul` dispatches PQ2_0 at m >= the qmm threshold to the tiled
//!      kernel; its bf16 output matches the host-decoded f64 reference.
//!   2. The tiled output matches the qmv output (fired directly) — same decoded
//!      weights, so they agree to a tight bf16 bound.
//!   3. Padded rows (m not a multiple of the row tile) do not corrupt the real
//!      rows: m = 24 forces bm = 16 and a padded tile of 32.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

const PQ2_0_FILE: &str = "linear/quant_pq2_0.metal";
const QMV_GROUP: [u32; 3] = [32, 2, 1];
const BLOCK_BYTES: usize = 34;
const BLOCK: usize = 128;
const QS_OFFSET: usize = 2;

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    let b = &bytes[i * 2..i * 2 + 2];
    f32::from_bits((u32::from(b[1]) << 24) | (u32::from(b[0]) << 16))
}

/// An fp16 bit pattern for a small positive normal scale (~0.015..0.03), built
/// directly so the test needs no f32->f16 rounding: exponent 9 or 10 (2^-6 or
/// 2^-5) with a random mantissa. The kernel and the host reference both decode
/// THIS pattern, so they agree regardless of its exact value.
fn scale_half_bits(seed: u64) -> u16 {
    let exp = 9u16 + u16::from(noise(seed) & 1); // 9 or 10
    let man = u16::from(noise(seed ^ 0x33)) | (u16::from(noise(seed ^ 0x77) & 0x3) << 8);
    (exp << 10) | (man & 0x3ff)
}

/// Exact fp16 -> f32 for a finite positive normal half (the only kind
/// `scale_half_bits` makes): `2^(exp-15) * (1 + man/1024)`.
fn f16_to_f32(bits: u16) -> f32 {
    let exp = i32::from((bits >> 10) & 0x1f);
    let man = f32::from(bits & 0x3ff) / 1024.0;
    (2.0f32).powi(exp - 15) * (1.0 + man)
}

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

/// One 34-byte PQ2_0 block: a leading fp16 scale then 32 code bytes (4 positional
/// 2-bit codes each). Returns the block bytes and the 128 decoded f32 weights so
/// the host reference rides along with the bytes.
fn make_block(seed: u64) -> ([u8; BLOCK_BYTES], [f32; BLOCK]) {
    let dbits = scale_half_bits(seed);
    let mut blk = [0u8; BLOCK_BYTES];
    blk[0..2].copy_from_slice(&dbits.to_le_bytes());
    for byte in 0..32 {
        blk[QS_OFFSET + byte] = noise(seed ^ ((byte as u64).wrapping_mul(0x100)));
    }
    let dh = f16_to_f32(dbits); // the scale as actually stored (half)
    let mut w = [0.0f32; BLOCK];
    for (j, wj) in w.iter_mut().enumerate() {
        let code = (blk[QS_OFFSET + (j >> 2)] >> ((j & 3) * 2)) & 0x03;
        *wj = (f32::from(code) - 1.0) * dh;
    }
    (blk, w)
}

#[test]
fn a_tiled_quaternary_qmm_answers_the_qmv_every_case() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    // (K, N, m). N multiples of 16/32/64 so a column tile divides them; K a whole
    // number of 128-blocks (> 32 blocks exercises the BK=32 lane wrap). m includes
    // 24 (padding case: bm=16, padded=32) and the bm rungs 16/32/64/128.
    let grid = [
        (2048usize, 256usize, 16usize),
        (2048, 256, 24),
        (5120, 512, 32),
        (5120, 512, 64),
        (6144, 128, 64),
        (4096, 320, 128),
    ];

    let mut cases = 0usize;
    for (k, n, m) in grid {
        one_case(&device, &handles, &pipelines, k, n, m);
        cases += 1;
    }
    eprintln!("tiled PQ2_0 qmm answered the qmv and the host reference on {cases} cases");
}

fn one_case(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    k: usize,
    n: usize,
    m: usize,
) {
    assert!(k.is_multiple_of(BLOCK), "K must be whole 128-blocks");
    let blocks_per_row = k / BLOCK;
    // capacity: pad m up to the row tile the dispatch will choose (<= 64 rung, or
    // 128). A generous capacity (next multiple of 64, min 64) keeps padded rows
    // in-bounds for both act and y.
    let cap = m.div_ceil(64).max(1) * 64;

    // Codes + host-decoded weights.
    let mut codes = vec![0u8; n * blocks_per_row * BLOCK_BYTES];
    let mut hw = vec![0.0f32; n * k];
    for r in 0..n {
        for c in 0..blocks_per_row {
            let (blk, w) = make_block((r as u64) ^ ((c as u64) << 20));
            let at = (r * blocks_per_row + c) * BLOCK_BYTES;
            codes[at..at + BLOCK_BYTES].copy_from_slice(&blk);
            hw[r * k + c * BLOCK..r * k + c * BLOCK + BLOCK].copy_from_slice(&w);
        }
    }
    // Activation: `cap` rows so padded reads/writes stay in-bounds; rows >= m are
    // never checked.
    let mut x = vec![0u8; cap * k * 2];
    for (at, pair) in x.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        pair.copy_from_slice(&bf16(0.02 * (f32::from(noise(at as u64) % 16) - 8.0)));
    }

    // f64 reference: exact host_decoded_W . x over the real m rows.
    let mut reference = vec![0.0f64; m * n];
    for (v, row) in reference.chunks_exact_mut(n).enumerate() {
        for (r, slot) in row.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += f64::from(hw[r * k + p]) * f64::from(bf16_at(&x, v * k + p));
            }
            *slot = acc;
        }
    }

    let mut codes_b = Buffer::zeroed(device, codes.len() as u64).expect("codes");
    codes_b.write(0, &codes).expect("write codes");
    let mut x_b = Buffer::zeroed(device, x.len() as u64).expect("x");
    x_b.write(0, &x).expect("write x");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let hc = bind(&codes_b);
    let hx = bind(&x_b);

    let (mi, ni, ki) = (
        u32::try_from(m).unwrap(),
        u32::try_from(n).unwrap(),
        u32::try_from(k).unwrap(),
    );

    // Arm 1: the dispatched tiled qmm (bf16 out).
    let yt_b = Buffer::zeroed(device, (cap * n * 2) as u64).expect("yt");
    let hyt = bind(&yt_b);
    let bank = Bank {
        codes: Tensor::new(hc, ni, ki, Dtype::Pq2_0),
        scales: Tensor::new(hc, ni, 1, Dtype::Bf16),
        biases: None,
        group: 128,
        bits: 2,
        mpp_codes: None,
    };
    let act = Tensor::new(hx, mi, ki, Dtype::Bf16);
    let yt = Tensor::new(hyt, mi, ni, Dtype::Bf16);
    let none = |_: u32, _: u32| None;
    let scratch = quant::Scratch {
        precast: &none,
        partials: &none,
    };
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    quant::matmul(&sink, act, bank, yt, scratch, u32::try_from(cap).unwrap())
        .expect("the dispatched tiled launch");
    frame.commit().expect("commit");
    let yt_raw = handles.read(hyt, (m * n * 2) as u64).expect("read yt");

    // Arm 2: the qmv fired directly (bf16 out), same codes and activation.
    let yv_b = Buffer::zeroed(device, (m * n * 2) as u64).expect("yv");
    let hyv = bind(&yv_b);
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    sink.fire(
        Fire::at(PQ2_0_FILE, "pq2_0_qmv_bfloat16").apply(Grid::of(
            quant::qmv_grid("qmv", i32::try_from(m).unwrap(), i32::try_from(n).unwrap())
                .expect("grid"),
            QMV_GROUP,
        )),
        &[
            Tensor::new(hc, ni, ki, Dtype::Pq2_0).arg(),
            Tensor::new(hx, mi, ki, Dtype::Bf16).arg(),
            Tensor::new(hyv, mi, ni, Dtype::Bf16).arg_mut(),
            i32::try_from(k).unwrap().arg(),
            i32::try_from(n).unwrap().arg(),
        ],
    )
    .expect("the qmv launch");
    frame.commit().expect("commit");
    let yv_raw = handles.read(hyv, (m * n * 2) as u64).expect("read yv");

    let mut worst_ref = 0.0f64;
    let mut worst_qmv = 0.0f64;
    for v in 0..m {
        for r in 0..n {
            let got = f64::from(bf16_at(&yt_raw, v * n + r));
            let want = reference[v * n + r];
            let scale = want.abs().max(got.abs()).max(0.1);
            worst_ref = worst_ref.max((want - got).abs() / scale);

            let qmv = f64::from(bf16_at(&yv_raw, v * n + r));
            let s2 = qmv.abs().max(got.abs()).max(0.1);
            worst_qmv = worst_qmv.max((qmv - got).abs() / s2);
        }
    }
    // bf16 output rounds at ~2^-8 (3.9e-3); 1.5e-2 covers that plus the f32
    // accumulation-order difference between the scalar qmv and the simdgroup mma.
    assert!(
        worst_ref <= 1.5e-2,
        "tiled qmm K={k} N={n} m={m} drifts {worst_ref:.2e} from the host reference"
    );
    assert!(
        worst_qmv <= 1.5e-2,
        "tiled qmm K={k} N={n} m={m} disagrees {worst_qmv:.2e} with the qmv"
    );
    eprintln!(
        "  K={k:>5} N={n:>4} m={m:>3}: vs host {worst_ref:.2e}, vs qmv {worst_qmv:.2e} (<=1.5e-2)"
    );
}
