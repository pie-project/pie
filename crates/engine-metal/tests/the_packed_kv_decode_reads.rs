#![cfg(target_vendor = "apple")]

//! C2c-1 — the DECODE sdpa read over a PACKED 4-bit KV cache.
//!
//! C2b proved the quantizing WRITE path leaves a self-consistent 4-bit cache.
//! This proves the matching READ: the decode sdpa kernel, dispatched on the
//! pool's `KvU4` dtype, byte-indexes each 130-byte block, unpacks the lane's
//! offset-binary nibbles (`q = nibble - 8`), rescales by the block's inline fp16
//! scale, and attends with the SAME math as the bf16 decode.
//!
//! ISOLATING DEQUANT FROM QUANT LOSS. Comparing the packed decode against a
//! bf16 decode on the ORIGINAL K/V would fold in the 4-bit quantization error
//! and could hide a read bug behind it. Instead the reference is a bf16 decode
//! over the K/V UNPACKED FROM THE PACKED CACHE: the exact values the read kernel
//! should reconstruct. If the read dequantizes + attends correctly the two agree
//! to a bf16-rounding-tight tolerance; a wrong offset or nibble parity shows up
//! as a large deviation (that is a bug to surface, not a tolerance to loosen).
//! The packed-vs-original deviation is printed too — that IS the 4-bit quant
//! error, expected small-but-nonzero.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::attn::{self, DecodePlan};
use kernels_metal::{KvPool, Tensor};
use model_ir::Dtype;

const BLOCK: usize = 256;
const PACKED: usize = 130; // 128 nibble bytes + one fp16 scale

const KV_HEADS: usize = 2;
const Q_HEADS: usize = 4; // gqa = 2
const HEAD_DIM: usize = 256;
const N_KV: usize = 20; // cached key/value tokens (positions 0..N_KV)
const PAGE_SIZE: u32 = 32;
const PAGES: u32 = 1;

// --- deterministic K/V/Q data ------------------------------------------------

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E5E_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit01(at: u64) -> f64 {
    (f64::from(noise(at)) + 1.0) / (f64::from(u32::MAX) + 2.0)
}

fn gaussian(at: u64) -> f32 {
    let u1 = unit01(at);
    let u2 = unit01(at ^ 0x9999_5A5A_1357_2468);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Small Gaussians with a couple of modest per-block outliers — enough to make
/// the 4-bit quant error visible in the FYI number, small enough that the
/// bf16-rounding gap the isolate check measures stays tight.
fn plane(rows: usize, width: usize, salt: u64) -> Vec<f32> {
    const SPIKES: usize = 2;
    let mut v: Vec<f32> = (0..(rows * width) as u64)
        .map(|i| gaussian(i ^ salt))
        .collect();
    for blk in 0..(rows * width) / BLOCK {
        let base = blk * BLOCK;
        for k in 0..SPIKES {
            let key =
                (blk as u64).wrapping_mul(0x100_0193) ^ (k as u64).wrapping_mul(0x9E37) ^ salt;
            let pos = (noise(key) as usize) % BLOCK;
            let mag = 3.0 + 2.0 * (unit01(key ^ 0xBEEF) as f32);
            let sign = if noise(key ^ 0xF00D) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            v[base + pos] = sign * mag;
        }
    }
    v
}

/// bf16-EXACT plane: every value is `q * S` for an integer code `q in [-7, 7]`
/// and a power-of-two scale `S = 0.25 = 2^-2`, with a `+7*S` and a `-7*S` planted
/// in EVERY 256-block. Then per block `absmax = 7*S`, so the codec's fp16
/// `scale = absmax/7 = S` is exactly the same power of two, quantization is
/// LOSSLESS (`round(q*S / S) = q`), and every dequant value `q*S` (<=3 mantissa
/// bits, |q*S| <= 1.75) is representable exactly in bf16. The packed kernel's
/// fp32 dequant and the bf16 reference therefore see BIT-IDENTICAL K/V — the
/// bf16-rounding residual the random case reasons about is removed, so the two
/// decodes must agree to fp32 accumulation noise. `q` varies deterministically
/// per (block, element) for coverage.
fn exact_plane(rows: usize, width: usize, salt: u64) -> Vec<f32> {
    const S: f32 = 0.25; // 2^-2, exact in fp16 and bf16
    let mut v = vec![0.0f32; rows * width];
    for blk in 0..(rows * width) / BLOCK {
        let base = blk * BLOCK;
        for d in 0..BLOCK {
            let key =
                (blk as u64).wrapping_mul(0x100_0193) ^ (d as u64).wrapping_mul(0x9E37) ^ salt;
            let q = (noise(key) % 15) as i32 - 7; // [-7, 7]
            v[base + d] = q as f32 * S;
        }
        // Pin absmax = 7*S in every block so scale is exactly S.
        v[base] = 7.0 * S;
        v[base + 1] = -7.0 * S;
    }
    v
}

// --- bf16 / fp16 bit helpers -------------------------------------------------

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&f| f32_to_bf16_bits(f).to_le_bytes())
        .collect()
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn bf16_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Decode an IEEE binary16 bit pattern to f32 (exact — every f16 is an f32).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if (bits >> 15) & 1 == 1 { -1.0f32 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    if exp == 0 {
        sign * f32::from(mant) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + f32::from(mant) / 1024.0) * 2f32.powi(i32::from(exp) - 15)
    }
}

/// The fp32 K/V a correct read reconstructs from one packed 130-byte block:
/// dequant = (nibble - 8) * fp16_scale.
fn unpack_block(bytes: &[u8]) -> [f32; BLOCK] {
    let scale_bits = u16::from(bytes[128]) | (u16::from(bytes[129]) << 8);
    let scale = f16_bits_to_f32(scale_bits);
    let mut out = [0.0f32; BLOCK];
    for byte_i in 0..128 {
        let byte = bytes[byte_i];
        out[2 * byte_i] = (i32::from(byte & 0xf) - 8) as f32 * scale;
        out[2 * byte_i + 1] = (i32::from(byte >> 4) - 8) as f32 * scale;
    }
    out
}

// --- rig ---------------------------------------------------------------------

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    keep: Vec<Buffer>,
}

impl Rig {
    fn open() -> Option<Self> {
        Some(Self {
            device: Context::bind().ok()?,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
            keep: Vec::new(),
        })
    }

    fn bf16(&mut self, data: &[f32]) -> u32 {
        self.bytes(&bf16_bytes(data))
    }

    fn u32s(&mut self, data: &[u32]) -> u32 {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        self.bytes(&bytes)
    }

    fn i32s(&mut self, data: &[i32]) -> u32 {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        self.bytes(&bytes)
    }

    fn u8s(&mut self, data: &[u8]) -> u32 {
        self.bytes(data)
    }

    fn bytes(&mut self, bytes: &[u8]) -> u32 {
        let mut b = Buffer::zeroed(&self.device, bytes.len().max(1) as u64).expect("a buffer");
        if !bytes.is_empty() {
            b.write(0, bytes).expect("write");
        }
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.push(b);
        h
    }

    /// A zeroed plane of `bytes` bytes; returns the handle and byte length.
    fn zeroed(&mut self, bytes: u64) -> (u32, u64) {
        let b = Buffer::zeroed(&self.device, bytes).expect("a plane");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.push(b);
        (h, bytes)
    }

    fn fire(&self, f: impl FnOnce(&Sink<'_>)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        f(&sink);
        frame.commit().expect("commit");
    }
}

// --- pool builders -----------------------------------------------------------

const CELLS: u32 = PAGES * PAGE_SIZE;
const WIDTH: u32 = (KV_HEADS * HEAD_DIM) as u32;

fn packed_pool(rig: &mut Rig) -> (KvPool, u32, u32, u64) {
    let (keys, kbytes) = rig.zeroed((CELLS as usize * KV_HEADS * PACKED) as u64);
    let (values, _) = rig.zeroed((CELLS as usize * KV_HEADS * PACKED) as u64);
    let (pool, _, _) = pool_over(rig, keys, values, Dtype::KvU4);
    (pool, keys, values, kbytes)
}

fn bf16_pool(rig: &mut Rig) -> (KvPool, u32) {
    let (keys, _) = rig.zeroed((CELLS as usize * WIDTH as usize * 2) as u64);
    let (values, _) = rig.zeroed((CELLS as usize * WIDTH as usize * 2) as u64);
    let (pool, _, _) = pool_over(rig, keys, values, Dtype::Bf16);
    (pool, keys)
}

fn pool_over(rig: &mut Rig, keys: u32, values: u32, dtype: Dtype) -> (KvPool, u32, u32) {
    let ppi = rig.u32s(&[0u32]); // one page, id 0
    let ppp = rig.u32s(&[0u32, 1]); // request 0 owns pages [0, 1)
    (
        KvPool {
            keys: Tensor::new(keys, CELLS, WIDTH, dtype),
            values: Tensor::new(values, CELLS, WIDTH, dtype),
            page_indices: Tensor::new(ppi, 1, 1, Dtype::U32),
            page_indptr: Tensor::new(ppp, 2, 1, Dtype::U32),
            page_size: PAGE_SIZE as i32,
            seq_stride: WIDTH as u64,
            head_stride: HEAD_DIM as u64,
        },
        ppi,
        ppp,
    )
}

/// Write the seeded K/V (bf16 tensors) into a pool via the real append op.
fn write_kv(rig: &Rig, pool: &KvPool, kt: Tensor, vt: Tensor, wpt: Tensor, wot: Tensor) {
    rig.fire(|s| {
        attn::kv_append(s, kt, vt, pool, wpt, wot).expect("kv_append launch");
    });
}

/// Decode one output plane over `pool`; returns rows*Q_HEADS*HEAD_DIM f32.
#[allow(clippy::too_many_arguments)]
fn decode_over(
    rig: &Rig,
    pool: &KvPool,
    qt: Tensor,
    plan: &DecodePlan,
    sm_scale: f32,
    rows: usize,
) -> Vec<f32> {
    let out_elems = rows * Q_HEADS * HEAD_DIM;
    let (ho, obytes) = {
        // out buffer bound fresh each call
        let b = Buffer::zeroed(&rig.device, (out_elems * 2) as u64).expect("out");
        let h = rig.handles.bind(&b, 0, b.bytes()).expect("out handle");
        // leak into a Box so it outlives the fire (handles index it)
        Box::leak(Box::new(b));
        (h, (out_elems * 2) as u64)
    };
    let ot = Tensor::new(ho, rows as u32, (Q_HEADS * HEAD_DIM) as u32, Dtype::Bf16);
    rig.fire(|s| {
        attn::decode(s, qt, plan, pool, None, HEAD_DIM as u32, sm_scale, ot)
            .expect("decode launch");
    });
    bf16_floats(&rig.handles.read(ho, obytes).expect("read out"))
}

fn deviation(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (f64::from(*x) - f64::from(*y)).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / a.len() as f64)
}

/// The per-case measurement: pack the seeded K/V, decode over the packed cache,
/// over a bf16 cache holding the K/V unpacked FROM the packed cache (the isolate
/// reference), and over a bf16 cache holding the ORIGINAL K/V (the quant-error
/// reference). Returns ((isolate max, isolate mean), (quant max, quant mean)).
fn run_case(rig: &mut Rig, label: &str, k_src: &[f32], v_src: &[f32]) -> ((f64, f64), (f64, f64)) {
    // ---- seed the write tables ----------------------------------------------
    let hk = rig.bf16(k_src);
    let hv = rig.bf16(v_src);
    let w_page = rig.u32s(&[0u32; N_KV]);
    let w_off = rig.u32s(&(0..N_KV as u32).collect::<Vec<_>>());
    let kt = Tensor::new(hk, N_KV as u32, WIDTH, Dtype::Bf16);
    let vt = Tensor::new(hv, N_KV as u32, WIDTH, Dtype::Bf16);
    let wpt = Tensor::new(w_page, N_KV as u32, 1, Dtype::U32);
    let wot = Tensor::new(w_off, N_KV as u32, 1, Dtype::U32);

    // ---- the packed cache + the original bf16 cache (same seeded K/V) --------
    let (packed, keys_h, values_h, kbytes) = packed_pool(rig);
    write_kv(rig, &packed, kt, vt, wpt, wot);
    let (orig_bf16, _) = bf16_pool(rig);
    write_kv(rig, &orig_bf16, kt, vt, wpt, wot);

    // ---- reference cache: the K/V UNPACKED from the packed cache -------------
    // Read the packed bytes back, dequantize each (slot, head) 256-block to fp32,
    // and lay it out as a [N_KV, KV_HEADS*HEAD_DIM] plane, then append that into a
    // fresh bf16 pool. Decoding over this pool is what a correct packed read must
    // reproduce (up to bf16 storage rounding of the dequantized values).
    let keys_raw = rig.handles.read(keys_h, kbytes).expect("read packed keys");
    let values_raw = rig
        .handles
        .read(values_h, kbytes)
        .expect("read packed values");
    let mut k_deq = vec![0.0f32; N_KV * KV_HEADS * HEAD_DIM];
    let mut v_deq = vec![0.0f32; N_KV * KV_HEADS * HEAD_DIM];
    for i in 0..N_KV {
        for h in 0..KV_HEADS {
            let slot = i; // page 0, offset i
            let base = (slot * KV_HEADS + h) * PACKED;
            let kb = unpack_block(&keys_raw[base..base + PACKED]);
            let vb = unpack_block(&values_raw[base..base + PACKED]);
            let row = i * KV_HEADS * HEAD_DIM + h * HEAD_DIM;
            k_deq[row..row + HEAD_DIM].copy_from_slice(&kb);
            v_deq[row..row + HEAD_DIM].copy_from_slice(&vb);
        }
    }
    let hkd = rig.bf16(&k_deq);
    let hvd = rig.bf16(&v_deq);
    let ktd = Tensor::new(hkd, N_KV as u32, WIDTH, Dtype::Bf16);
    let vtd = Tensor::new(hvd, N_KV as u32, WIDTH, Dtype::Bf16);
    let (unpacked_bf16, _) = bf16_pool(rig);
    write_kv(rig, &unpacked_bf16, ktd, vtd, wpt, wot);

    // ---- decode plan: a few decode queries at varied positions, request 0 ----
    let positions = [3i32, 8, 14, (N_KV - 1) as i32];
    let rows = positions.len();
    let hpos = rig.i32s(&positions);
    let hreq = rig.i32s(&vec![0i32; rows]);
    let hmen = rig.u8s(&vec![0u8; rows]); // mask disabled → full causal to position
    let hmask = rig.u8s(&vec![0u8; rows]); // never read while mask disabled
    let plan = DecodePlan {
        positions: Tensor::new(hpos, rows as u32, 1, Dtype::I32),
        request_of_token: Tensor::new(hreq, rows as u32, 1, Dtype::I32),
        mask: Tensor::new(hmask, rows as u32, 1, Dtype::U8),
        mask_enabled: Tensor::new(hmen, rows as u32, 1, Dtype::U8),
        mask_stride: 1,
    };

    // seeded queries, bf16-rounded (same across cases via the fixed seed)
    let q_src: Vec<f32> = (0..(rows * Q_HEADS * HEAD_DIM) as u64)
        .map(|i| 0.5 * gaussian(i ^ 0x0071))
        .collect();
    let hq = rig.bf16(&q_src);
    let qt = Tensor::new(hq, rows as u32, (Q_HEADS * HEAD_DIM) as u32, Dtype::Bf16);
    let sm_scale = (HEAD_DIM as f32).sqrt().recip();

    // ---- the three decodes --------------------------------------------------
    let out_packed = decode_over(rig, &packed, qt, &plan, sm_scale, rows);
    let out_unpacked = decode_over(rig, &unpacked_bf16, qt, &plan, sm_scale, rows);
    let out_orig = decode_over(rig, &orig_bf16, qt, &plan, sm_scale, rows);

    // ---- (isolate) packed read vs bf16 decode on the unpacked-from-packed K/V
    let iso = deviation(&out_packed, &out_unpacked);
    // ---- (FYI) packed read vs bf16 decode on the ORIGINAL K/V = 4-bit quant err
    let quant = deviation(&out_packed, &out_orig);

    eprintln!(
        "[{label:>10} isolate] packed decode vs bf16-decode-on-unpacked-from-packed: \
         max={:.3e} mean={:.3e}",
        iso.0, iso.1
    );
    eprintln!(
        "[{label:>10} quant  ] packed decode vs bf16-decode-on-original K/V:         \
         max={:.3e} mean={:.3e}",
        quant.0, quant.1
    );
    (iso, quant)
}

#[test]
fn the_packed_kv_decode_reads() {
    eprintln!("=== C2c-1: dequantizing decode read for packed 4-bit KV ===");
    let Some(mut rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    eprintln!("device: {}", rig.device.name());

    // ---- CASE 1: realistic random K/V (off-grid) ----------------------------
    // Off-grid Gaussians with modest spikes: quantization is lossy (the FYI quant
    // number is meaty) and the isolate deviation is bounded by bf16 STORAGE
    // rounding of the reconstructed values — the output is a softmax-convex
    // combination of V, so it cannot deviate by more than the bf16 ulp of the
    // biggest V it mixes. Its telltale of correctness is the MEAN: a wrong offset
    // (q vs q+8), swapped nibble parity, or a mis-indexed lane corrupts the
    // reconstructed K/V across every block and drives the mean up by orders of
    // magnitude. Assert the mean is tight AND the whole isolate error sits far
    // below the real 4-bit quant error (the read reconstructs the INTENDED
    // values, it is not merely landing inside quantization noise).
    let ((r_iso_max, r_iso_mean), (r_q_max, _)) = run_case(
        &mut rig,
        "random",
        &plane(N_KV, KV_HEADS * HEAD_DIM, 0x0C2C),
        &plane(N_KV, KV_HEADS * HEAD_DIM, 0x0DEC),
    );
    assert!(
        r_iso_mean < 2.0e-3,
        "random: the packed decode parts from the bf16 decode on the SAME (unpacked) \
         K/V by mean {r_iso_mean:.3e} — a systematic dequant/offset/parity bug, not \
         bf16 rounding"
    );
    assert!(
        r_iso_max < 3.0e-2,
        "random: the packed decode's worst element parts by {r_iso_max:.3e}, above \
         the bf16 storage-rounding ceiling — surface it as a read bug, do not loosen"
    );
    assert!(
        r_iso_max * 6.0 < r_q_max,
        "random: the isolate deviation {r_iso_max:.3e} is not clearly below the 4-bit \
         quant error {r_q_max:.3e}; the read must reconstruct the intended K/V, not \
         merely land within quantization noise"
    );
    assert!(
        r_q_max.is_finite() && r_q_max > 0.0,
        "random: the 4-bit quant error should be finite and nonzero, got {r_q_max:.3e}"
    );

    // ---- CASE 2: bf16-EXACT K/V (on-grid) — the unambiguous exactness proof --
    // Every seed is q*S for q in [-7,7], S = 2^-2, with ±7*S planted per block.
    // Quantization is LOSSLESS and every dequant value is bf16-exact, so the
    // packed kernel's fp32 dequant and the bf16 reference read BIT-IDENTICAL K/V.
    // The two decode paths run the SAME fp32 body over the SAME inputs, so they
    // must agree to fp32-accumulation noise (expected ~0). This removes the
    // bf16-rounding residual entirely: a TIGHT max here is a direct proof the
    // dequant (offset, parity, scale, byte-index) is exact. Because the quant is
    // lossless, the "quant" reference is ALSO bit-identical here — so its number
    // is ~0 too, and there is no lossy error to separate from (the random case
    // above carries that contrast).
    let ((e_iso_max, e_iso_mean), (e_q_max, _)) = run_case(
        &mut rig,
        "bf16-exact",
        &exact_plane(N_KV, KV_HEADS * HEAD_DIM, 0x0E7A),
        &exact_plane(N_KV, KV_HEADS * HEAD_DIM, 0x0E7B),
    );
    assert!(
        e_iso_max < 1.0e-5,
        "bf16-exact: the packed decode parts from the bf16 decode on BIT-IDENTICAL \
         (on-grid) K/V by max {e_iso_max:.3e} — the dequant is NOT exact (a real \
         offset/parity/scale/index bug), not a tolerance to loosen"
    );
    eprintln!(
        "[bf16-exact] isolate max {e_iso_max:.3e} mean {e_iso_mean:.3e}, lossless-quant \
         max {e_q_max:.3e} — dequant is exact on-grid"
    );
    eprintln!(
        "=== C2c-1 done: the packed 4-bit KV decode read dequantizes + attends correctly ==="
    );
}
