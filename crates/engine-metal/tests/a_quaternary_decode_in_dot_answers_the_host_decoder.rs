#![cfg(target_vendor = "apple")]

//! M2b: the Metal PQ2_0 (positional 2-bit, `g128_u2_f16_n`) decode-in-dot kernel
//! must match the M2a host decoder — which is itself bit-exact against the fork's
//! own `dequantize_row_pq2_0`. This is the PTQ1_0-analog (`a_ternary_decode_in_dot
//! _answers_the_host_decoder`): prove the Metal read path reproduces the proven
//! host reference on REAL blocks.
//!
//! The oracle is the frozen fixture `checkpoint/src/codec/pq2_0_fixture.rs`: real
//! 34-byte blocks lifted from Ternary-Bonsai-2-27B-PQ2_0.gguf, each with the f32
//! bit patterns the fork's dequant produces (`expect_bits`). We include it
//! verbatim so the host oracle values ride along with the blocks.
//!
//! Two checks:
//!   1. Bit-exact read-back via one-hot probes. Feed the kernel the identity
//!      (x = e_i) so `y[row] = decoded_W[row][i]`; with a one-hot activation only
//!      one term accumulates, so an exact dequant lands an exact value. We assert
//!      the recovered weights equal the fixture's `expect_bits` BIT-FOR-BIT on
//!      every element of every block. Output is f32 here so the full 32-bit
//!      decoded value survives (a bf16 output would truncate it).
//!   2. General matmul correctness with a random activation, against the host-
//!      decoded weights. The f32-out kernel is held to a tight bound (only f32
//!      accumulation order differs from the f64 reference); the dispatched
//!      bf16-out path is held to a bf16-output bound.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

// The M2a oracle: `ForkBlock { tensor, block_index, bytes: [u8;34], expect_bits:
// [u32;128] }` and `FORK_BLOCKS`, generated from the real GGUF + the fork dequant.
include!("../../checkpoint/src/codec/pq2_0_fixture.rs");

const PQ2_0_FILE: &str = "linear/quant_pq2_0.metal";
const QMV_GROUP: [u32; 3] = [32, 2, 1];
const BLOCK_BYTES: usize = 34;
const BLOCK: usize = 128;

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

fn f32_at(bytes: &[u8], i: usize) -> f32 {
    let b = &bytes[i * 4..i * 4 + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    let b = &bytes[i * 2..i * 2 + 2];
    f32::from_bits((u32::from(b[1]) << 24) | (u32::from(b[0]) << 16))
}

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

#[test]
fn a_quaternary_decode_in_dot_answers_the_host_decoder_every_case() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    the_one_hot_probe_reads_back_the_host_decoder_bit_exact(&device, &handles, &pipelines);
    a_random_activation_answers_the_host_decoded_weights(&device, &handles, &pipelines);
}

/// Check 1: one block per row (K = 128), identity activation. `y[i*N + r]` is the
/// dot of row `r` with `e_i`, i.e. the single decoded weight `decoded_W[r][i]`.
/// Assert it equals the fixture's `expect_bits[r][i]` bit-for-bit across every
/// element of every block.
fn the_one_hot_probe_reads_back_the_host_decoder_bit_exact(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
) {
    let blocks: Vec<&ForkBlock> = FORK_BLOCKS.iter().collect();
    let n = blocks.len(); // output rows (one fixture block each)
    let k = BLOCK; // one 128-weight block per row
    let m = BLOCK; // the 128 one-hot probes, as 128 activation vectors

    // codes: N rows, row_bytes = 34 (one block).
    let mut codes = vec![0u8; n * BLOCK_BYTES];
    for (r, b) in blocks.iter().enumerate() {
        codes[r * BLOCK_BYTES..r * BLOCK_BYTES + BLOCK_BYTES].copy_from_slice(&b.bytes);
    }
    // x: the 128x128 identity in bf16 (vector i is e_i).
    let mut x = vec![0u8; m * k * 2];
    for i in 0..m {
        let pair = &mut x[(i * k + i) * 2..(i * k + i) * 2 + 2];
        pair.copy_from_slice(&bf16(1.0));
    }

    let mut codes_b = Buffer::zeroed(device, codes.len() as u64).expect("codes");
    codes_b.write(0, &codes).expect("write codes");
    let mut x_b = Buffer::zeroed(device, x.len() as u64).expect("x");
    x_b.write(0, &x).expect("write x");
    let y_b = Buffer::zeroed(device, (m * n * 4) as u64).expect("y");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hc, hx, hy) = (bind(&codes_b), bind(&x_b), bind(&y_b));

    let (mi, ni, ki) = (
        i32::try_from(m).unwrap(),
        i32::try_from(n).unwrap(),
        i32::try_from(k).unwrap(),
    );
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    sink.fire(
        Fire::at(PQ2_0_FILE, "pq2_0_qmv_bfloat16_f32").apply(Grid::of(
            quant::qmv_grid("m2b.readback", mi, ni).expect("grid"),
            QMV_GROUP,
        )),
        &[
            Tensor::new(hc, ni.unsigned_abs(), ki.unsigned_abs(), Dtype::Pq2_0).arg(),
            Tensor::new(hx, mi.unsigned_abs(), ki.unsigned_abs(), Dtype::Bf16).arg(),
            Tensor::new(hy, mi.unsigned_abs(), ni.unsigned_abs(), Dtype::F32).arg_mut(),
            ki.arg(),
            ni.arg(),
        ],
    )
    .expect("the readback launch");
    frame.commit().expect("commit");

    let y = handles.read(hy, (m * n * 4) as u64).expect("read y");

    let mut real = 0usize;
    let mut synthetic = 0usize;
    let mut checked = 0usize;
    for (r, b) in blocks.iter().enumerate() {
        if b.tensor.starts_with("synthetic:") {
            synthetic += 1;
        } else {
            real += 1;
        }
        for i in 0..BLOCK {
            let got = f32_at(&y, i * n + r).to_bits();
            let want = b.expect_bits[i];
            assert_eq!(
                got, want,
                "{} block {} element {}: metal {:#010x} != host {:#010x}",
                b.tensor, b.block_index, i, got, want,
            );
            checked += 1;
        }
    }
    assert!(real >= 20, "want the spread of real blocks, got {real}");
    eprintln!(
        "readback: {checked} elements bit-exact vs the host decoder \
         across {real} real + {synthetic} synthetic blocks"
    );
}

/// Check 2: a wider matrix (many blocks per row, so multiple lanes and the
/// `bk += 32` wrap are exercised) against a random activation. The f64 reference
/// is the exact `host_decoded_W · x` (both operands are exactly representable).
/// The f32-out kernel differs only by f32 accumulation order (tight bound); the
/// dispatched bf16-out path additionally rounds the result to bf16.
fn a_random_activation_answers_the_host_decoded_weights(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
) {
    let n = 16usize; // output rows
    let blocks_per_row = 40usize; // K = 5120; 40 > 32 exercises the lane wrap
    let k = blocks_per_row * BLOCK;
    let m = 3usize; // activation vectors

    let total = FORK_BLOCKS.len();
    // Row r, block-column c takes fixture block (r*blocks_per_row + c) % total.
    let src = |r: usize, c: usize| &FORK_BLOCKS[(r * blocks_per_row + c) % total];

    let mut codes = vec![0u8; n * blocks_per_row * BLOCK_BYTES];
    for r in 0..n {
        for c in 0..blocks_per_row {
            let at = (r * blocks_per_row + c) * BLOCK_BYTES;
            codes[at..at + BLOCK_BYTES].copy_from_slice(&src(r, c).bytes);
        }
    }
    // Random bf16 activation in [-1, 1)-ish.
    let mut x = vec![0u8; m * k * 2];
    for (at, pair) in x.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let v = 0.02 * (f32::from(noise(at as u64) % 16) - 8.0);
        pair.copy_from_slice(&bf16(v));
    }

    // f64 reference: exact host_decoded_W . x.
    let mut reference = vec![0.0f64; m * n];
    for v in 0..m {
        for r in 0..n {
            let mut acc = 0.0f64;
            for c in 0..blocks_per_row {
                let b = src(r, c);
                for p in 0..BLOCK {
                    let w = f64::from(f32::from_bits(b.expect_bits[p]));
                    let xv = f64::from(bf16_at(&x, v * k + c * BLOCK + p));
                    acc += w * xv;
                }
            }
            reference[v * n + r] = acc;
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
        i32::try_from(m).unwrap(),
        i32::try_from(n).unwrap(),
        i32::try_from(k).unwrap(),
    );

    // Arm A: f32-out kernel fired directly — the tight bound.
    let yf_b = Buffer::zeroed(device, (m * n * 4) as u64).expect("yf");
    let hyf = bind(&yf_b);
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    sink.fire(
        Fire::at(PQ2_0_FILE, "pq2_0_qmv_bfloat16_f32").apply(Grid::of(
            quant::qmv_grid("m2b.general.f32", mi, ni).expect("grid"),
            QMV_GROUP,
        )),
        &[
            Tensor::new(hc, ni.unsigned_abs(), ki.unsigned_abs(), Dtype::Pq2_0).arg(),
            Tensor::new(hx, mi.unsigned_abs(), ki.unsigned_abs(), Dtype::Bf16).arg(),
            Tensor::new(hyf, mi.unsigned_abs(), ni.unsigned_abs(), Dtype::F32).arg_mut(),
            ki.arg(),
            ni.arg(),
        ],
    )
    .expect("the f32 launch");
    frame.commit().expect("commit");
    let yf = handles.read(hyf, (m * n * 4) as u64).expect("read yf");

    let mut worst_f32 = 0.0f64;
    for v in 0..m {
        for r in 0..n {
            let got = f64::from(f32_at(&yf, v * n + r));
            let want = reference[v * n + r];
            let scale = want.abs().max(got.abs()).max(1.0);
            worst_f32 = worst_f32.max((want - got).abs() / scale);
        }
    }
    // K=5120 f32 accumulations of values ~O(0.01..0.02) (the `+2` code doubles the
    // weight but not the order of magnitude): relative drift ~ K * 2^-24 ~ 3e-4
    // worst case; 2e-3 is a safe, tight ceiling that a real defect blows.
    assert!(
        worst_f32 <= 2e-3,
        "f32-out 2-bit matmul drifts {worst_f32:.2e} from the host-decoded reference"
    );

    // Arm B: the dispatched bf16-out path — proves `act_x_wt` routes PQ2_0 to the
    // 2-bit kernel and handles the single-plane inline-scale bank.
    let y_b = Buffer::zeroed(device, (m * n * 2) as u64).expect("y");
    let hy = bind(&y_b);
    let bank = Bank {
        codes: Tensor::new(hc, ni.unsigned_abs(), ki.unsigned_abs(), Dtype::Pq2_0),
        // Single-plane inline scale: no scales/biases plane. `scales` is required
        // by the struct but the PQ2_0 path never reads it; alias it to codes.
        scales: Tensor::new(hc, ni.unsigned_abs(), 1, Dtype::Bf16),
        biases: None,
        group: 128,
        bits: 2,
    };
    let act = Tensor::new(hx, mi.unsigned_abs(), ki.unsigned_abs(), Dtype::Bf16);
    let y = Tensor::new(hy, mi.unsigned_abs(), ni.unsigned_abs(), Dtype::Bf16);
    let none = |_: u32, _: u32| None;
    let scratch = quant::Scratch {
        precast: &none,
        partials: &none,
    };
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    quant::matmul(&sink, act, bank, y, scratch, mi.unsigned_abs()).expect("the dispatched launch");
    frame.commit().expect("commit");
    let yb = handles.read(hy, (m * n * 2) as u64).expect("read y");

    let mut worst_bf16 = 0.0f64;
    for v in 0..m {
        for r in 0..n {
            let got = f64::from(bf16_at(&yb, v * n + r));
            let want = reference[v * n + r];
            let scale = want.abs().max(got.abs()).max(1.0);
            worst_bf16 = worst_bf16.max((want - got).abs() / scale);
        }
    }
    // bf16 keeps 8 mantissa bits, so the final store rounds at ~2^-8 = 3.9e-3;
    // 8e-3 covers that plus the f32 accumulation drift.
    assert!(
        worst_bf16 <= 8e-3,
        "dispatched bf16-out 2-bit matmul drifts {worst_bf16:.2e} from the reference"
    );
    eprintln!(
        "general: f32-out worst {worst_f32:.2e} (<=2e-3), dispatched bf16-out worst \
         {worst_bf16:.2e} (<=8e-3) over {m}x{n}, K={k}"
    );
}
