//! Vulkan M1b: the Vulkan PTQ1_0 (ternary, `g128_t3_f16_n`) decode-in-dot kernel
//! must match the host decoder — itself bit-exact against the fork's own
//! `dequantize_row_ptq1_0`. Ported 1:1 from the engine-metal oracle test; the
//! numeric contract, fixture, and bounds are identical. Only the device/kernel
//! surface differs (Vulkan Slang kernel `quant/ptq1_0.slang`, groupshared
//! reduction). Runs on any Vulkan device incl. lavapipe; skips with no device.
//!
//! Two checks:
//!   1. Bit-exact read-back via one-hot probes (identity activation → each
//!      decoded weight), asserted against the fixture's `expect_bits` bit-for-bit.
//!   2. General matmul vs the host-decoded weights (f64 reference): the f32-out
//!      kernel to a tight bound, the dispatched bf16-out path to a bf16 bound.

use engine_vulkan::encode::Sink;
use engine_vulkan::{Buffer, Context, DeviceBoot, Handles, Pipelines};
use kernels_vulkan::encode::{Arg, Encode, Fire};
use kernels_vulkan::linear::quant;
use kernels_vulkan::tensor::{Bank, Tensor};
use model_ir::Dtype;

// The M1a oracle: `ForkBlock { tensor, block_index, bytes: [u8;28], expect_bits:
// [u32;128] }` and `FORK_BLOCKS`, generated from the real GGUF + the fork dequant.
include!("../../checkpoint/src/codec/ptq1_0_fixture.rs");

const PTQ1_0_FILE: &str = "quant/ptq1_0.slang";
const QMV_F32: &str = "ptq1_0_qmv_bfloat16_f32";
const PTQ1_0_GROUP: [u32; 3] = [32, 8, 1];
const PTQ1_0_RLANES: u32 = 8;
const BLOCK_BYTES: usize = 28;
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

/// Fire the f32-out ternary qmv directly, mirroring `quant::ptq1_0_matmul`'s
/// launch geometry: one workgroup owns `PTQ1_0_RLANES` output rows for one
/// activation vector; push constants are `{out_vec_size=n, in_vec_size=k, vecs=m}`.
fn fire_f32(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    hc: u32,
    hx: u32,
    hy: u32,
    m: u32,
    n: u32,
    k: u32,
) {
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    sink.fire(
        Fire::at(PTQ1_0_FILE, QMV_F32)
            .groups([n.div_ceil(PTQ1_0_RLANES), m, 1])
            .group(PTQ1_0_GROUP),
        &[
            Tensor::new(hc, n, k, Dtype::Ptq1_0).arg(),
            Tensor::new(hx, m, k, Dtype::Bf16).arg(),
            Tensor::new(hy, m, n, Dtype::F32).arg_mut(),
            n.arg(),
            k.arg(),
            m.arg(),
        ],
    )
    .expect("the f32 launch");
    frame.commit().expect("commit");
}

#[test]
fn a_ternary_decode_in_dot_answers_the_host_decoder_every_case() {
    let Ok(device) = Context::bind(&DeviceBoot::default()) else {
        eprintln!("not asked: no Vulkan device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    the_one_hot_probe_reads_back_the_host_decoder_bit_exact(&device, &handles, &pipelines);
    a_random_activation_answers_the_host_decoded_weights(&device, &handles, &pipelines);
}

fn the_one_hot_probe_reads_back_the_host_decoder_bit_exact(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
) {
    let blocks: Vec<&ForkBlock> = FORK_BLOCKS.iter().collect();
    let n = blocks.len(); // output rows (one fixture block each)
    let k = BLOCK; // one 128-weight block per row
    let m = BLOCK; // the 128 one-hot probes, as 128 activation vectors

    let mut codes = vec![0u8; n * BLOCK_BYTES];
    for (r, b) in blocks.iter().enumerate() {
        codes[r * BLOCK_BYTES..r * BLOCK_BYTES + BLOCK_BYTES].copy_from_slice(&b.bytes);
    }
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

    fire_f32(
        device, handles, pipelines, hc, hx, hy, m as u32, n as u32, k as u32,
    );

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
                "{} block {} element {}: vulkan {:#010x} != host {:#010x}",
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
    let src = |r: usize, c: usize| &FORK_BLOCKS[(r * blocks_per_row + c) % total];

    let mut codes = vec![0u8; n * blocks_per_row * BLOCK_BYTES];
    for r in 0..n {
        for c in 0..blocks_per_row {
            let at = (r * blocks_per_row + c) * BLOCK_BYTES;
            codes[at..at + BLOCK_BYTES].copy_from_slice(&src(r, c).bytes);
        }
    }
    let mut x = vec![0u8; m * k * 2];
    for (at, pair) in x.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let v = 0.02 * (f32::from(noise(at as u64) % 16) - 8.0);
        pair.copy_from_slice(&bf16(v));
    }

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

    // Arm A: f32-out kernel fired directly — the tight bound.
    let yf_b = Buffer::zeroed(device, (m * n * 4) as u64).expect("yf");
    let hyf = bind(&yf_b);
    fire_f32(
        device, handles, pipelines, hc, hx, hyf, m as u32, n as u32, k as u32,
    );
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
    assert!(
        worst_f32 <= 2e-3,
        "f32-out ternary matmul drifts {worst_f32:.2e} from the host-decoded reference"
    );

    // Arm B: the dispatched bf16-out path — proves `act_x_wt` routes PTQ1_0 to
    // the ternary kernel and handles the single-plane inline-scale bank.
    let y_b = Buffer::zeroed(device, (m * n * 2) as u64).expect("y");
    let hy = bind(&y_b);
    let bank = Bank {
        codes: Tensor::new(hc, n as u32, k as u32, Dtype::Ptq1_0),
        // Single-plane inline scale: no scales/biases plane. `scales` is required
        // by the struct but the PTQ1_0 path never reads it; alias it to codes.
        scales: Tensor::new(hc, n as u32, 1, Dtype::Bf16),
        biases: None,
        group: 128,
        bits: 2,
    };
    let act = Tensor::new(hx, m as u32, k as u32, Dtype::Bf16);
    let y = Tensor::new(hy, m as u32, n as u32, Dtype::Bf16);
    let none = |_: u32, _: u32| None;
    let scratch = quant::Scratch { precast: &none };
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(device, &frame, pipelines, handles);
    quant::matmul(&sink, act, bank, y, scratch, m as u32).expect("the dispatched launch");
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
    assert!(
        worst_bf16 <= 8e-3,
        "dispatched bf16-out ternary matmul drifts {worst_bf16:.2e} from the reference"
    );
    eprintln!(
        "general: f32-out worst {worst_f32:.2e} (<=2e-3), dispatched bf16-out worst \
         {worst_bf16:.2e} (<=8e-3) over {m}x{n}, K={k}"
    );
}
