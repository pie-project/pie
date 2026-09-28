#![cfg(target_vendor = "apple")]

//! C0 — the core Phase-C hypothesis at the kernel level: rotating a KV-like
//! tensor with a Hadamard *before* low-bit quantization reduces reconstruction
//! error (incoherence processing).
//!
//! The mechanism, plainly: a KV block is mostly small values with a few large
//! outlier spikes. A per-block absmax quantizer sizes its step from the largest
//! magnitude, so one big spike inflates the step and the small values collapse
//! toward zero — most of the block is lost. The Hadamard is orthonormal, so it
//! preserves the block's L2 energy but *spreads* each spike evenly across all
//! `head_dim` channels (a spike of size M becomes ~M/sqrt(head_dim) everywhere).
//! The absmax shrinks, the step shrinks, and the whole block is represented far
//! better. Because H is self-inverse (H·H = I), we recover the original layout
//! by applying it a second time after dequantization, and — again by
//! orthonormality — the reconstruction error in the original domain equals the
//! quantization error measured in the rotated domain.
//!
//! This TEST-ONLY file simulates the (not-yet-existent) KV codec with a pure
//! host-side per-block symmetric absmax quantizer, drives the real Metal
//! `elementwise.hadamard` op for the rotation, and quantifies the reduction at
//! 4-bit and 3-bit, head_dim 256 and 128, on outlier-heavy data. It also runs
//! the same experiment on plain Gaussian data to show — honestly — that the
//! rotation helps *because* of outliers, and does little when there are none.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::pointwise;
use model_ir::Dtype;

/// Deterministic integer hash (same mixer as the sibling Hadamard test), so the
/// data is reproducible run to run with no RNG state.
fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5e5e_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

/// A uniform in (0, 1), never exactly 0 (so the log in Box-Muller is finite).
fn unit01(at: u64) -> f64 {
    (f64::from(noise(at)) + 1.0) / (f64::from(u32::MAX) + 2.0)
}

/// A deterministic standard-normal sample via Box-Muller.
fn gaussian(at: u64) -> f32 {
    let u1 = unit01(at);
    let u2 = unit01(at ^ 0x9999_5A5A_1357_2468);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Drive the real Metal `elementwise.hadamard` over `rows x width` f32 values
/// held row-major, in place (pure transform, no sign diagonal), and read the
/// result back. Mirrors the harness in `the_hadamard_transform_agrees.rs`.
fn hadamard_metal(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    rows: u32,
    width: u32,
    block: u32,
    data: &[f32],
) -> Vec<f32> {
    let bytes = u64::from(rows) * u64::from(width) * 4;
    let mut buf = Buffer::zeroed(device, bytes).expect("a buffer");
    buf.write(0, &f32_bytes(data)).expect("write x");
    let handle = handles.bind(&buf, 0, buf.bytes()).expect("a handle");
    let x = Tensor::new(handle, rows, width, Dtype::F32);
    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(device, &frame, pipelines, handles);
        pointwise::hadamard(&sink, x, block, None).expect("the launch");
        frame.commit().expect("the commit");
    }
    f32_floats(&handles.read(handle, bytes).expect("read x"))
}

/// The stand-in KV codec: per contiguous `block`-vector, a symmetric absmax
/// quantizer at `bits` bits. `scale = absmax / ((1<<(bits-1)) - 1)`, then
/// `dq = round(clamp(x/scale, -lim, lim)) * scale`. Pure host Rust — this is the
/// low-bit path we are testing the rotation against, not a production codec.
fn quantize_rowmajor(x: &[f32], block: usize, bits: u32) -> Vec<f32> {
    let lim = ((1i64 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0f32; x.len()];
    for (bi, chunk) in x.chunks_exact(block).enumerate() {
        let base = bi * block;
        let absmax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if absmax == 0.0 {
            out[base..base + block].copy_from_slice(chunk);
            continue;
        }
        let scale = absmax / lim;
        for (j, &v) in chunk.iter().enumerate() {
            let q = (v / scale).round().clamp(-lim, lim);
            out[base + j] = q * scale;
        }
    }
    out
}

/// Mean-squared error and max-absolute error of a reconstruction, in f64.
fn errors(orig: &[f32], recon: &[f32]) -> (f64, f64) {
    let mut sse = 0.0f64;
    let mut maxabs = 0.0f64;
    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let d = f64::from(o) - f64::from(r);
        sse += d * d;
        maxabs = maxabs.max(d.abs());
    }
    (sse / orig.len() as f64, maxabs)
}

/// Outlier-heavy KV-like data: every element is a small Gaussian, then a few
/// large spikes are planted at deterministic positions in each block — the
/// classic KV outlier-channel shape that defeats a plain absmax quantizer.
fn outlier_data(rows: usize, head_dim: usize, salt: u64) -> Vec<f32> {
    const SPIKES: usize = 3;
    let mut v = vec![0.0f32; rows * head_dim];
    for r in 0..rows {
        let base = r * head_dim;
        for c in 0..head_dim {
            v[base + c] = gaussian((base + c) as u64 ^ salt);
        }
        for k in 0..SPIKES {
            let key = (r as u64).wrapping_mul(0x100_0193) ^ (k as u64).wrapping_mul(0x9E37) ^ salt;
            let pos = (noise(key) as usize) % head_dim;
            // A spike of magnitude ~20..40, sign from a fresh draw.
            let mag = 20.0 + 20.0 * (unit01(key ^ 0xBEEF) as f32);
            let sign = if noise(key ^ 0xF00D) & 1 == 0 { 1.0 } else { -1.0 };
            v[base + pos] = sign * mag;
        }
    }
    v
}

/// Plain Gaussian data — no planted outliers. The control case where rotation
/// should give little to no benefit.
fn uniform_data(rows: usize, head_dim: usize, salt: u64) -> Vec<f32> {
    (0..(rows * head_dim) as u64)
        .map(|i| gaussian(i ^ salt))
        .collect()
}

/// Run both paths (unrotated and rotated) over one dataset and return
/// `((mse_plain, max_plain), (mse_rot, max_rot))`.
fn compare_paths(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    rows: u32,
    head_dim: u32,
    bits: u32,
    data: &[f32],
) -> ((f64, f64), (f64, f64)) {
    let block = head_dim as usize;

    // Unrotated: quantize -> dequantize the raw blocks.
    let plain_recon = quantize_rowmajor(data, block, bits);
    let plain = errors(data, &plain_recon);

    // Rotated: H -> quantize/dequantize in the rotated domain -> H again to undo.
    let rotated = hadamard_metal(device, handles, pipelines, rows, head_dim, head_dim, data);
    let rotated_q = quantize_rowmajor(&rotated, block, bits);
    let rot_recon = hadamard_metal(device, handles, pipelines, rows, head_dim, head_dim, &rotated_q);
    let rot = errors(data, &rot_recon);

    (plain, rot)
}

#[test]
fn the_rotated_kv_quantizes_better() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();

    const ROWS: u32 = 64;

    // --- Sanity: the pure transform is self-inverse, so H(H(x)) == x. This is
    // the "undo" step the rotated path leans on; if it drifts, every number
    // below is suspect. f32, tight tolerance.
    for &head_dim in &[256u32, 128] {
        let data = uniform_data(ROWS as usize, head_dim as usize, 0xDEAD ^ u64::from(head_dim));
        let once = hadamard_metal(&device, &handles, &pipelines, ROWS, head_dim, head_dim, &data);
        let twice = hadamard_metal(&device, &handles, &pipelines, ROWS, head_dim, head_dim, &once);
        let (_, max_rt) = errors(&data, &twice);
        eprintln!("[sanity] head_dim={head_dim}: H(H(x)) round-trip max-abs err = {max_rt:.3e}");
        assert!(
            max_rt < 1e-3,
            "pure Hadamard must be self-inverse within 1e-3 (head_dim={head_dim}), got {max_rt:.3e}"
        );
    }

    eprintln!();
    eprintln!("=== C0: rotated vs unrotated low-bit reconstruction (MSE / max-abs) ===");

    for &head_dim in &[256u32, 128] {
        for &bits in &[4u32, 3] {
            // --- OUTLIER-HEAVY: the case incoherence processing is meant for.
            let odata = outlier_data(ROWS as usize, head_dim as usize, 0x0117 ^ u64::from(head_dim));
            let ((mse_p, max_p), (mse_r, max_r)) =
                compare_paths(&device, &handles, &pipelines, ROWS, head_dim, bits, &odata);
            let ratio = mse_r / mse_p;
            eprintln!(
                "[outlier] head_dim={head_dim:>3} {bits}-bit | plain MSE={mse_p:.5e} max={max_p:.4} \
                 || rot MSE={mse_r:.5e} max={max_r:.4} || MSE ratio rot/plain = {ratio:.4}"
            );
            // The claim: rotation reduces reconstruction error on outlier data,
            // in every corner. The margin is GRADUATED, and the printed ratios
            // above quantify it: dramatic at 4-bit (ratio ~0.12 at head_dim 256,
            // ~0.17 at 128 — a 6-8x MSE cut) and smaller but real at 3-bit
            // (~0.56 at 256, ~0.82 at 128). Two knobs explain the falloff: fewer
            // levels (3-bit's lim=3 vs 4-bit's lim=7) and a smaller block both
            // give the rotated domain less room to represent the now-spread
            // energy. 0.9 is the weakest honest margin across (bits x head_dim);
            // it asserts the reduction always holds without pretending the 3-bit
            // small-head win is as large as the 4-bit one.
            assert!(
                mse_r < 0.9 * mse_p,
                "outlier head_dim={head_dim} {bits}-bit: expected rotated MSE below plain \
                 (rot={mse_r:.5e}, plain={mse_p:.5e}, ratio={ratio:.4})"
            );
            // Max-abs error should improve too (the collapsed small values recover).
            assert!(
                max_r < max_p,
                "outlier head_dim={head_dim} {bits}-bit: expected rotated max-abs below plain \
                 (rot={max_r:.4}, plain={max_p:.4})"
            );

            // --- UNIFORM (control): no outliers, rotation should NOT help much.
            let udata = uniform_data(ROWS as usize, head_dim as usize, 0x2222 ^ u64::from(head_dim));
            let ((umse_p, umax_p), (umse_r, umax_r)) =
                compare_paths(&device, &handles, &pipelines, ROWS, head_dim, bits, &udata);
            let uratio = umse_r / umse_p;
            eprintln!(
                "[uniform] head_dim={head_dim:>3} {bits}-bit | plain MSE={umse_p:.5e} max={umax_p:.4} \
                 || rot MSE={umse_r:.5e} max={umax_r:.4} || MSE ratio rot/plain = {uratio:.4}"
            );
            // Honesty guard: on non-outlier data the rotation is roughly neutral,
            // and specifically must not be dramatically worse.
            assert!(
                umse_r < 2.0 * umse_p,
                "uniform head_dim={head_dim} {bits}-bit: rotation should be ~neutral, not much \
                 worse (rot={umse_r:.5e}, plain={umse_p:.5e}, ratio={uratio:.4})"
            );
        }
    }

    eprintln!("=== C0 done: incoherence processing reduces low-bit KV error on outlier data ===");
}
