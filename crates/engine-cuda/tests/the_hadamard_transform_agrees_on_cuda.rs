#![cfg(feature = "cuda")]

//! `elementwise.hadamard` on CUDA is a blockwise butterfly FWHT: the row's last
//! dim is cut into contiguous `block`-vectors (a power of two) and each is
//! turned by the normalized Sylvester Hadamard matrix H, whose entries are
//! `(-1)^popcount(i & j) / sqrt(block)`. The fast Walsh-Hadamard transform's
//! natural (Sylvester) ordering is exactly H, so the pure transform is
//! symmetric and orthonormal: `H . H = I`.
//!
//! When a +-1 sign diagonal S is supplied it multiplies the activation on load,
//! before the butterfly -- the Randomized Hadamard `H.S`. S and H do not
//! commute, so `H.S` is NOT self-inverse; the signed path is checked only by
//! agreement with the host reference.
//!
//! This checks the CUDA output against a host reference across both the warp
//! (block <= 256) and shared-memory (block >= 512) pipelines, pins the exact
//! matrix entries with a one-hot orientation probe, and confirms the round trip
//! for the pure transform. The f32 path is checked to tight tolerance and the
//! bf16 instantiation to a loose one. Gated to skip when no CUDA device is
//! present; the lead runs it on the GPU.

use engine_cuda::device::{Buffer, Context};
use kernels_cuda::Tensor;
use kernels_cuda::elemwise::fwht;
use model_ir::Dtype;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E5E_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (f64::from(noise(at)) / f64::from(u32::MAX)) as f32 * 2.0 - 1.0
}

/// A deterministic +-1 sign vector of the given width.
fn signs_of(width: usize, salt: u64) -> Vec<f32> {
    (0..width as u64)
        .map(|i| if noise(i ^ salt) & 1 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Round-to-nearest-even f32 -> bf16, matching `device.cuh`'s `f32_to_bf16`.
fn f32_to_bf16(f: f32) -> u16 {
    let b = f.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    ((b + rounding) >> 16) as u16
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&f| f32_to_bf16(f).to_le_bytes())
        .collect()
}

fn bf16_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// The host reference: each contiguous `block`-vector is optionally scaled by
/// the +-1 diagonal S (which repeats block-wise across the flattened row) and
/// then left-multiplied by the normalized Sylvester Hadamard matrix. Signs are
/// indexed by flat offset, matching the kernel's `p % signs_width`. Accumulated
/// in f64 for a clean gold.
fn blockwise_h_signed(x: &[f32], block: usize, signs: Option<&[f32]>) -> Vec<f32> {
    let inv = 1.0f64 / (block as f64).sqrt();
    let sw = signs.map_or(0, <[f32]>::len);
    let mut out = vec![0.0f32; x.len()];
    for (b, chunk) in x.chunks_exact(block).enumerate() {
        let base = b * block;
        let signed: Vec<f64> = (0..block)
            .map(|j| {
                let s = signs.map_or(1.0, |sg| f64::from(sg[(base + j) % sw]));
                s * f64::from(chunk[j])
            })
            .collect();
        for i in 0..block {
            let mut acc = 0.0f64;
            for (j, &v) in signed.iter().enumerate() {
                let sign = if (i & j).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                acc += sign * v;
            }
            out[base + i] = (acc * inv) as f32;
        }
    }
    out
}

/// Fire the CUDA op over `rows x width` f32 values held row-major, in place, and
/// read the result back. An optional +-1 sign diagonal is bound as a second
/// buffer.
fn run_f32(
    context: &Context,
    rows: u32,
    width: u32,
    block: u32,
    data: &[f32],
    signs: Option<&[f32]>,
) -> Vec<f32> {
    let bytes = data.len() * 4;
    let mut buf = Buffer::zeroed(bytes).expect("a device buffer");
    buf.write(0, &f32_bytes(data)).expect("write x");

    let sign_hold = signs.map(|s| {
        let mut sbuf = Buffer::zeroed(s.len() * 4).expect("a sign buffer");
        sbuf.write(0, &f32_bytes(s)).expect("write signs");
        let stensor = Tensor::new(sbuf.ptr(), 1, s.len() as u32, Dtype::F32);
        (sbuf, stensor)
    });
    let signs_tensor = sign_hold.as_ref().map(|(_, t)| *t);

    let mut x = Tensor::new(buf.ptr(), rows, width, Dtype::F32);
    fwht::hadamard(context.ctx(), &mut x, block, signs_tensor).expect("the launch");
    context.synchronize().expect("the sync");

    let mut out = vec![0u8; bytes];
    buf.read(0, &mut out).expect("read x");
    f32_floats(&out)
}

/// The bf16 counterpart: the input is rounded to bf16 before upload and the
/// answer is read back as bf16, converted to f32.
fn run_bf16(
    context: &Context,
    rows: u32,
    width: u32,
    block: u32,
    data: &[f32],
    signs: Option<&[f32]>,
) -> Vec<f32> {
    let bytes = data.len() * 2;
    let mut buf = Buffer::zeroed(bytes).expect("a device buffer");
    buf.write(0, &bf16_bytes(data)).expect("write x");

    let sign_hold = signs.map(|s| {
        let mut sbuf = Buffer::zeroed(s.len() * 2).expect("a sign buffer");
        sbuf.write(0, &bf16_bytes(s)).expect("write signs");
        let stensor = Tensor::new(sbuf.ptr(), 1, s.len() as u32, Dtype::Bf16);
        (sbuf, stensor)
    });
    let signs_tensor = sign_hold.as_ref().map(|(_, t)| *t);

    let mut x = Tensor::new(buf.ptr(), rows, width, Dtype::Bf16);
    fwht::hadamard(context.ctx(), &mut x, block, signs_tensor).expect("the launch");
    context.synchronize().expect("the sync");

    let mut out = vec![0u8; bytes];
    buf.read(0, &mut out).expect("read x");
    bf16_floats(&out)
}

fn close(got: f32, want: f32, tol: f32, at: &str) {
    let bound = tol * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= bound,
        "{at}: got {got}, want {want} (|delta| = {})",
        (got - want).abs()
    );
}

/// Agreement of the pure (unsigned) f32 transform for a rectangle.
fn agrees(context: &Context, rows: u32, width: u32, block: u32) {
    let n = u64::from(rows) * u64::from(width);
    let salt = (u64::from(rows) << 32) ^ (u64::from(width) << 8) ^ u64::from(block);
    let x: Vec<f32> = (0..n).map(|at| unit(at ^ salt)).collect();
    let got = run_f32(context, rows, width, block, &x, None);
    let want = blockwise_h_signed(&x, block as usize, None);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        close(
            g,
            w,
            1e-4,
            &format!("[{rows}x{width}] block {block} element {i}"),
        );
    }
}

/// Agreement of the signed transform `H.(S.x)` for a rectangle.
fn agrees_signed(context: &Context, rows: u32, width: u32, block: u32, signs_width: usize) {
    let n = u64::from(rows) * u64::from(width);
    let salt = (u64::from(rows) << 32) ^ (u64::from(width) << 8) ^ u64::from(block);
    let x: Vec<f32> = (0..n).map(|at| unit(at ^ salt ^ 0x5157_u64)).collect();
    let signs = signs_of(signs_width, 0xF00D ^ u64::from(block));
    let got = run_f32(context, rows, width, block, &x, Some(&signs));
    let want = blockwise_h_signed(&x, block as usize, Some(&signs));
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        close(
            g,
            w,
            1e-4,
            &format!("signed [{rows}x{width}] block {block} sw {signs_width} element {i}"),
        );
    }
}

#[test]
fn the_hadamard_transform_agrees_on_cuda() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let context = Context::bind(0, None).expect("a CUDA context");

    // Agreement against the host reference across both pipelines: block 64, 128
    // and 256 ride the warp variant; 512 and 1024 ride the shared-memory
    // variant. Shapes cover one block, several blocks, and a batch of rows.
    for &block in &[64u32, 128, 256, 512, 1024] {
        agrees(&context, 1, block, block); // one block
        agrees(&context, 1, block * 4, block); // several blocks
        agrees(&context, 5, block * 3, block); // a batch
    }
    // The 5120-wide Qwen 27B hidden width, several blocks of 512, batched.
    agrees(&context, 6, 5120, 512);

    // Signed path `H.(S.x)` against the host reference. A full-width diagonal
    // (signs_width == width) and a block-wide diagonal that repeats block-wise
    // across the row are both exercised, on both pipelines.
    for &block in &[128u32, 256, 512, 1024] {
        agrees_signed(&context, 4, block * 3, block, (block * 3) as usize);
        agrees_signed(&context, 4, block * 3, block, block as usize);
    }

    // Orientation probe: a one-hot input picks out one column of H. Because H is
    // symmetric this is also row k, and its exact signed entries are pinned -- a
    // transposed or mis-strided matrix would not reproduce them. Tested at each
    // block size, and in the SECOND block of a two-block row so striding is
    // exercised too.
    for &block in &[64u32, 128, 256, 512, 1024] {
        let bl = block as usize;
        let inv = 1.0f32 / (block as f32).sqrt();
        let k = 3usize;
        let mut x = vec![0.0f32; bl * 2];
        x[bl + k] = 1.0;
        let got = run_f32(&context, 1, block * 2, block, &x, None);
        for (i, &g) in got.iter().take(bl).enumerate() {
            close(g, 0.0, 1e-4, &format!("probe block0 (n={block}) entry {i}"));
        }
        for i in 0..bl {
            let want = if (i & k).count_ones() & 1 == 1 {
                -inv
            } else {
                inv
            };
            close(
                got[bl + i],
                want,
                1e-4,
                &format!("probe block1 (n={block}) entry {i}"),
            );
        }
    }

    // Round trip: H . H = I for the PURE transform, so applying twice returns
    // the input. (Not tested for the signed transform, which is not an
    // involution.)
    for &block in &[64u32, 128, 256, 512, 1024] {
        let n = u64::from(block) * 9;
        let width = block * 9;
        let x: Vec<f32> = (0..n)
            .map(|at| unit(at ^ 0xABCD ^ u64::from(block)))
            .collect();
        let once = run_f32(&context, 1, width, block, &x, None);
        let twice = run_f32(&context, 1, width, block, &once, None);
        for (i, (&t, &orig)) in twice.iter().zip(x.iter()).enumerate() {
            close(
                t,
                orig,
                1e-4,
                &format!("round-trip (n={block}) element {i}"),
            );
        }
    }

    // The bf16 instantiation fires on both pipelines and agrees with the host
    // reference (over the bf16-rounded input) to a loose tolerance.
    for &block in &[256u32, 1024] {
        let width = block * 3;
        let n = 3u64 * u64::from(width);
        let x: Vec<f32> = (0..n)
            .map(|at| unit(at ^ 0xB16 ^ u64::from(block)))
            .collect();
        let rounded: Vec<f32> = x.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
        let got = run_bf16(&context, 3, width, block, &x, None);
        let want = blockwise_h_signed(&rounded, block as usize, None);
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            close(
                g,
                w,
                3e-2,
                &format!("bf16 [3x{width}] block {block} element {i}"),
            );
        }
    }
}
