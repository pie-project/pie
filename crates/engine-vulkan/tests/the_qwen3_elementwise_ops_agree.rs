//! The five elementwise ops the Ternary-Bonsai-2-27B (qwen_3) forward needs
//! that engine-vulkan newly dispatches, each checked by bit-close agreement
//! against an f64 host reference on the real device. One `#[test]` per op; each
//! skips cleanly when no Vulkan device is present.
//!
//! All five kernels are bf16-only (`dtype_dispatch!` offers a single `Bf16`
//! entry), so every activation rides a `Dtype::Bf16` tensor. Inputs are first
//! quantized to bf16 and decoded back, so the host reference reads exactly the
//! values the kernel reads; the only slack is the kernel's final round of the
//! f32 result back to bf16. The tolerance therefore tracks a bf16 ulp
//! (`2^-7` relative), far tighter than any convention or wiring bug would
//! survive, and matches the spirit of the hadamard test's `close`.
//!
//! Harness (device bind / Buffer / Handles / Sink / fire / read-back) is the
//! same one `the_hadamard_transform_agrees.rs` uses.

use engine_vulkan::encode::Sink;
use engine_vulkan::{Buffer, Context, DeviceBoot, Handles, Pipelines};
use kernels_vulkan::elemwise::{gate, norm, pointwise, rope};
use kernels_vulkan::tensor::Tensor;
use model_ir::Dtype;

// ---- deterministic inputs -------------------------------------------------

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E5E_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

/// A value in roughly [-2, 2), deterministic in `at`.
fn unit(at: u64) -> f32 {
    (f64::from(noise(at)) / f64::from(u32::MAX)) as f32 * 4.0 - 2.0
}

// ---- bf16, matching kernels/common/bf16.slang exactly ---------------------

fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return 0x7fc0; // NaN
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// Quantize a host `f32` field to its bf16 bytes (for upload) and to the f64
/// value the kernel actually sees (for the reference). Keeping both in step
/// removes input quantization from the comparison.
fn quantize(data: &[f32]) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    let mut seen = Vec::with_capacity(data.len());
    for &v in data {
        let b = f32_to_bf16(v);
        bytes.extend_from_slice(&b.to_le_bytes());
        seen.push(f64::from(bf16_to_f32(b)));
    }
    (bytes, seen)
}

fn read_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16_to_f32(u16::from_le_bytes(*c)))
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

// ---- agreement ------------------------------------------------------------

/// Bit-close for a bf16-landed result: within ~one bf16 ulp of the true f64
/// value, with an absolute floor for values near zero.
fn close(got: f32, want: f64, at: &str) {
    let tol = (1.0 / 128.0) * want.abs().max(1.0);
    let delta = (f64::from(got) - want).abs();
    assert!(delta <= tol, "{at}: got {got}, want {want} (|Δ| = {delta}, tol {tol})");
}

/// Bind a buffer of pre-encoded bytes and mint a tensor over it. The returned
/// `Buffer` must outlive the frame.
fn upload(
    device: &Context,
    handles: &Handles,
    bytes: &[u8],
    rows: u32,
    width: u32,
    dtype: Dtype,
) -> (Buffer, Tensor) {
    let mut buf = Buffer::zeroed(device, bytes.len() as u64).expect("a buffer");
    buf.write(0, bytes).expect("write");
    let handle = handles.bind(&buf, 0, buf.bytes()).expect("a handle");
    (buf, Tensor::new(handle, rows, width, dtype))
}

fn device() -> Option<(Context, Handles, Pipelines)> {
    let device = Context::bind(&DeviceBoot::default()).ok()?;
    eprintln!("device: {}", device.name());
    Some((device, Handles::new(), Pipelines::new()))
}

// ---- 1. add: z = x + y (out of place) -------------------------------------

#[test]
fn the_add_op_agrees() {
    let Some((device, handles, pipelines)) = device() else {
        eprintln!("not asked: no Vulkan device");
        return;
    };
    let (rows, width) = (3u32, 128u32);
    let n = (rows * width) as u64;
    let xf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x0A_DD0)).collect();
    let yf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x0A_DD1)).collect();
    let (xb, xq) = quantize(&xf);
    let (yb, yq) = quantize(&yf);

    let bytes = n * 2;
    let (_xbuf, x) = upload(&device, &handles, &xb, rows, width, Dtype::Bf16);
    let (_ybuf, y) = upload(&device, &handles, &yb, rows, width, Dtype::Bf16);
    let (zbuf, z) = upload(&device, &handles, &vec![0u8; bytes as usize], rows, width, Dtype::Bf16);
    let zhandle = z.buf;

    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        pointwise::add(&sink, x, y, z).expect("the launch");
        frame.commit().expect("the commit");
    }
    let got = read_bf16(&handles.read(zhandle, bytes).expect("read z"));
    let _ = zbuf;
    for (i, &g) in got.iter().enumerate() {
        close(g, xq[i] + yq[i], &format!("add element {i}"));
    }
    eprintln!("add: z == x + y over {rows}x{width} agrees");
}

// ---- 2. mul_scalar: x *= s (in place; s rounded to bf16 by the kernel) ----

#[test]
fn the_mul_scalar_op_agrees() {
    let Some((device, handles, pipelines)) = device() else {
        eprintln!("not asked: no Vulkan device");
        return;
    };
    let (rows, width) = (3u32, 128u32);
    let n = (rows * width) as u64;
    // A scalar the kernel first quantizes to bf16 (`PIE_LOAD(PIE_STORE(s))`);
    // the reference multiplies by that same quantized value.
    let s = 0.137_f32;
    let s_eff = f64::from(bf16_to_f32(f32_to_bf16(s)));
    let xf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x5CA1)).collect();
    let (xb, xq) = quantize(&xf);
    let bytes = n * 2;

    let (xbuf, x) = upload(&device, &handles, &xb, rows, width, Dtype::Bf16);
    let xhandle = x.buf;
    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        norm::mul_scalar(&sink, s, x).expect("the launch");
        frame.commit().expect("the commit");
    }
    let got = read_bf16(&handles.read(xhandle, bytes).expect("read x"));
    let _ = xbuf;
    for (i, &g) in got.iter().enumerate() {
        close(g, xq[i] * s_eff, &format!("mul_scalar element {i}"));
    }
    eprintln!("mul_scalar: x *= {s} over {rows}x{width} agrees");
}

// ---- 3. residual_add: y += x (in place) -----------------------------------

#[test]
fn the_residual_add_op_agrees() {
    let Some((device, handles, pipelines)) = device() else {
        eprintln!("not asked: no Vulkan device");
        return;
    };
    let (rows, width) = (4u32, 256u32);
    let n = (rows * width) as u64;
    let xf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x8E51)).collect();
    let yf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x8E52)).collect();
    let (xb, xq) = quantize(&xf);
    let (yb, yq) = quantize(&yf);
    let bytes = n * 2;

    let (_xbuf, x) = upload(&device, &handles, &xb, rows, width, Dtype::Bf16);
    let (ybuf, y) = upload(&device, &handles, &yb, rows, width, Dtype::Bf16);
    let yhandle = y.buf;
    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        norm::residual_add(&sink, x, y).expect("the launch");
        frame.commit().expect("the commit");
    }
    let got = read_bf16(&handles.read(yhandle, bytes).expect("read y"));
    let _ = ybuf;
    for (i, &g) in got.iter().enumerate() {
        close(g, yq[i] + xq[i], &format!("residual_add element {i}"));
    }
    eprintln!("residual_add: y += x over {rows}x{width} agrees");
}

// ---- 4. gate_sigmoid_mul: x *= sigmoid(gate) (in place) -------------------

#[test]
fn the_gate_sigmoid_mul_op_agrees() {
    let Some((device, handles, pipelines)) = device() else {
        eprintln!("not asked: no Vulkan device");
        return;
    };
    let (rows, width) = (4u32, 256u32);
    let n = (rows * width) as u64;
    let xf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x6A7E)).collect();
    let gf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x6A7F)).collect();
    let (xb, xq) = quantize(&xf);
    let (gb, gq) = quantize(&gf);
    let bytes = n * 2;

    let (xbuf, x) = upload(&device, &handles, &xb, rows, width, Dtype::Bf16);
    let (_gbuf, g) = upload(&device, &handles, &gb, rows, width, Dtype::Bf16);
    let xhandle = x.buf;
    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        gate::sigmoid_mul(&sink, g, x).expect("the launch");
        frame.commit().expect("the commit");
    }
    let got = read_bf16(&handles.read(xhandle, bytes).expect("read x"));
    let _ = xbuf;
    for (i, &g) in got.iter().enumerate() {
        let sig = 1.0 / (1.0 + (-gq[i]).exp());
        close(g, xq[i] * sig, &format!("gate_sigmoid_mul element {i}"));
    }
    eprintln!("gate_sigmoid_mul: x *= sigmoid(gate) over {rows}x{width} agrees");
}

// ---- 5. rope_partial (the sensitive one) ----------------------------------
//
// Convention derived from `kernels/rope/neox.slang` + `src/elemwise/rope.rs`.
// `rope::partial` calls `rotate(.., proportional = true)`, which fires the
// `neox_prop_mb_bf16` (PIE_PROP) entry. There, for each rotated pair index
// `i` in `0 .. rotary_dim/2`:
//
//     d        = 2 * i / head_dim                 (NOT /rotary_dim)
//     angle    = position * theta^(-d)            (base = theta.log2(), scale 1)
//     partner  = i + head_dim/2                   (NOT i + rotary_dim/2)
//     x[i]     = x[i]*cos - x[partner]*sin
//     x[part.] = x[i]*sin + x[partner]*cos
//
// So the rotated lanes are `[0, rotary/2)` paired with
// `[head_dim/2, head_dim/2 + rotary/2)`; every other lane is left untouched.
// This is the GPT-NeoX *proportional* half layout, which differs from a naive
// "pair i with i+rotary/2, exponent 2i/rotary" reading. Applied to both q and k.

#[allow(clippy::too_many_arguments)]
fn rope_reference(
    data: &[f64],
    positions: &[i32],
    rows: u32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
) -> Vec<f64> {
    let hd = head_dim as usize;
    let half = hd / 2;
    let pair = (rotary_dim / 2) as usize;
    // Mirror the kernel's f32 base exactly.
    let base = f64::from(theta.log2());
    let mut out = data.to_vec();
    for r in 0..rows as usize {
        let pos = f64::from(positions[r]);
        for h in 0..heads as usize {
            let head = r * heads as usize * hd + h * hd;
            for i in 0..pair {
                let d = 2.0 * i as f64 / hd as f64;
                let angle = pos * (2.0f64).powf(-d * base);
                let (s, c) = angle.sin_cos();
                let a = data[head + i];
                let b = data[head + i + half];
                out[head + i] = a * c - b * s;
                out[head + i + half] = a * s + b * c;
            }
        }
    }
    out
}

#[test]
fn the_rope_partial_op_agrees() {
    let Some((device, handles, pipelines)) = device() else {
        eprintln!("not asked: no Vulkan device");
        return;
    };
    let (heads, head_dim, rotary_dim) = (2u32, 16u32, 8u32);
    let width = heads * head_dim;
    let positions: Vec<i32> = vec![0, 1, 5, 37];
    let rows = positions.len() as u32;
    let n = u64::from(rows) * u64::from(width);
    let theta = 10_000.0_f32;

    let qf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x5107_0001)).collect();
    let kf: Vec<f32> = (0..n).map(|i| unit(i ^ 0x5107_0002)).collect();
    let (qb, qq) = quantize(&qf);
    let (kb, kq) = quantize(&kf);
    let bytes = n * 2;

    let (_qbuf, q) = upload(&device, &handles, &qb, rows, width, Dtype::Bf16);
    let (_kbuf, k) = upload(&device, &handles, &kb, rows, width, Dtype::Bf16);
    let (_pbuf, pos) = upload(
        &device,
        &handles,
        &i32_bytes(&positions),
        rows,
        1,
        Dtype::I32,
    );
    let qhandle = q.buf;
    let khandle = k.buf;
    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        rope::partial(&sink, q, k, pos, rotary_dim, head_dim, theta).expect("the launch");
        frame.commit().expect("the commit");
    }
    let got_q = read_bf16(&handles.read(qhandle, bytes).expect("read q"));
    let got_k = read_bf16(&handles.read(khandle, bytes).expect("read k"));

    let want_q = rope_reference(&qq, &positions, rows, heads, head_dim, rotary_dim, theta);
    let want_k = rope_reference(&kq, &positions, rows, heads, head_dim, rotary_dim, theta);

    for (i, &g) in got_q.iter().enumerate() {
        close(g, want_q[i], &format!("rope_partial q element {i}"));
    }
    for (i, &g) in got_k.iter().enumerate() {
        close(g, want_k[i], &format!("rope_partial k element {i}"));
    }

    // Dims outside the rotated halves must be left exactly as uploaded — the
    // kernel never writes lane `l` unless `l < rotary/2` or
    // `head_dim/2 <= l < head_dim/2 + rotary/2`.
    let half = (head_dim / 2) as usize;
    let pair = (rotary_dim / 2) as usize;
    let hd = head_dim as usize;
    for r in 0..rows as usize {
        for h in 0..heads as usize {
            let head = r * heads as usize * hd + h * hd;
            for lane in 0..hd {
                let rotated = lane < pair || (lane >= half && lane < half + pair);
                if !rotated {
                    let idx = head + lane;
                    assert_eq!(
                        f64::from(got_q[idx]),
                        qq[idx],
                        "rope_partial q lane {lane} (head {h}, row {r}) must be untouched"
                    );
                    assert_eq!(
                        f64::from(got_k[idx]),
                        kq[idx],
                        "rope_partial k lane {lane} (head {h}, row {r}) must be untouched"
                    );
                }
            }
        }
    }
    eprintln!(
        "rope_partial: proportional-neox halves (partner i+head_dim/2, exp 2i/head_dim) \
         agree on q and k across {rows} positions x {heads} heads"
    );
}
