#![cfg(target_vendor = "apple")]

//! C2c-2 — the TILED sdpa read over a PACKED 4-bit KV cache: prefill, the masked
//! prefill, and the prefill that also publishes a log-sum-exp (prefill_lse).
//!
//! C2b proved the quantizing WRITE path leaves a self-consistent 4-bit cache, and
//! C2c-1 proved the matching DECODE read. This proves the matching TILED read: the
//! prefill/masked/prefill_lse sdpa kernels, dispatched on the pool's `KvU4` dtype,
//! stage each cached (slot, kv_head) into shared memory by byte-indexing its
//! 130-byte block, unpacking the offset-binary nibbles (`q = nibble - 8`) and
//! rescaling by the block's inline fp16 scale — then run the SAME tile matmul,
//! softmax and lse as the bf16 tiled path.
//!
//! ISOLATING DEQUANT FROM QUANT LOSS (same discipline as C2c-1). Comparing the
//! packed read against a bf16 read on the ORIGINAL K/V would fold in the 4-bit
//! quantization error and could hide a read bug behind it. Instead the reference
//! is a bf16 read over the K/V UNPACKED FROM THE PACKED CACHE: the exact values
//! the staging load should reconstruct. A correct dequant + attention agrees with
//! it to a bf16-rounding-tight tolerance; a wrong offset, nibble parity, byte
//! index or scale shows up as a large deviation (a bug to surface, not a tolerance
//! to loosen). The packed-vs-original deviation is printed too — that IS the 4-bit
//! quant error, expected small-but-nonzero.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::attn::{self, PrefillPlan};
use kernels_metal::{KvPool, RaggedTensor, Tensor};
use model_ir::Dtype;

const KV_HEADS: usize = 2;
const Q_HEADS: usize = 4; // gqa = 2
const N_KV: usize = 20; // cached key/value tokens (positions 0..N_KV)
const PAGE_SIZE: u32 = 32;
const PAGES: u32 = 1;

/// The head_dims the packed read is proven at. The codec block IS the head_dim,
/// so this exercises the tiled read shaders stamped at 256 AND 128.
const HEAD_DIMS: [usize; 2] = [256, 128];

/// Packed bytes for one head at codec block `head_dim`: `head_dim/2` nibble bytes
/// plus one inline fp16 scale (130 at 256, 66 at 128).
fn packed_bytes(head_dim: usize) -> usize {
    head_dim / 2 + 2
}

/// The three tiled read ops this path serves — all one shared staging hook.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Prefill,
    Masked,
    PrefillLse,
}

impl Op {
    fn label(self) -> &'static str {
        match self {
            Op::Prefill => "prefill",
            Op::Masked => "masked",
            Op::PrefillLse => "prefill_lse",
        }
    }
}

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
fn plane(rows: usize, width: usize, block: usize, salt: u64) -> Vec<f32> {
    const SPIKES: usize = 2;
    let mut v: Vec<f32> = (0..(rows * width) as u64)
        .map(|i| gaussian(i ^ salt))
        .collect();
    for blk in 0..(rows * width) / block {
        let base = blk * block;
        for k in 0..SPIKES {
            let key =
                (blk as u64).wrapping_mul(0x100_0193) ^ (k as u64).wrapping_mul(0x9E37) ^ salt;
            let pos = (noise(key) as usize) % block;
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
/// fp32 dequant and the bf16 reference therefore stage BIT-IDENTICAL K/V — the
/// bf16-rounding residual the random case reasons about is removed, so the two
/// reads must agree to fp32 accumulation noise. `q` varies deterministically per
/// (block, element) for coverage.
fn exact_plane(rows: usize, width: usize, block: usize, salt: u64) -> Vec<f32> {
    const S: f32 = 0.25; // 2^-2, exact in fp16 and bf16
    let mut v = vec![0.0f32; rows * width];
    for blk in 0..(rows * width) / block {
        let base = blk * block;
        for d in 0..block {
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
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
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

#[allow(clippy::chunks_exact_to_as_chunks)]
fn f32_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

/// The fp32 K/V a correct read reconstructs from one packed `block`-element head
/// (`block/2` nibble bytes then a little-endian fp16 scale):
/// dequant = (nibble - 8) * fp16_scale.
fn unpack_block(bytes: &[u8], block: usize) -> Vec<f32> {
    let nib = block / 2;
    let scale_bits = u16::from(bytes[nib]) | (u16::from(bytes[nib + 1]) << 8);
    let scale = f16_bits_to_f32(scale_bits);
    let mut out = vec![0.0f32; block];
    for byte_i in 0..nib {
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

fn packed_pool(rig: &mut Rig, head_dim: usize) -> (KvPool, u32, u32, u64) {
    let (keys, kbytes) = rig.zeroed((CELLS as usize * KV_HEADS * packed_bytes(head_dim)) as u64);
    let (values, _) = rig.zeroed((CELLS as usize * KV_HEADS * packed_bytes(head_dim)) as u64);
    let pool = pool_over(rig, keys, values, Dtype::KvU4, head_dim);
    (pool, keys, values, kbytes)
}

fn bf16_pool(rig: &mut Rig, head_dim: usize) -> KvPool {
    let width = KV_HEADS * head_dim;
    let (keys, _) = rig.zeroed((CELLS as usize * width * 2) as u64);
    let (values, _) = rig.zeroed((CELLS as usize * width * 2) as u64);
    pool_over(rig, keys, values, Dtype::Bf16, head_dim)
}

fn pool_over(rig: &mut Rig, keys: u32, values: u32, dtype: Dtype, head_dim: usize) -> KvPool {
    let width = (KV_HEADS * head_dim) as u32;
    let ppi = rig.u32s(&[0u32]); // one page, id 0
    let ppp = rig.u32s(&[0u32, 1]); // request 0 owns pages [0, 1)
    KvPool {
        keys: Tensor::new(keys, CELLS, width, dtype),
        values: Tensor::new(values, CELLS, width, dtype),
        page_indices: Tensor::new(ppi, 1, 1, Dtype::U32),
        page_indptr: Tensor::new(ppp, 2, 1, Dtype::U32),
        page_size: PAGE_SIZE as i32,
        seq_stride: width as u64,
        head_stride: head_dim as u64,
    }
}

/// Write the seeded K/V (bf16 tensors) into a pool via the real append op.
fn write_kv(rig: &Rig, pool: &KvPool, kt: Tensor, vt: Tensor, wpt: Tensor, wot: Tensor) {
    rig.fire(|s| {
        attn::kv_append(s, kt, vt, pool, wpt, wot).expect("kv_append launch");
    });
}

// --- the query fire ----------------------------------------------------------

/// Query rows: a batch of prefill tokens for request 0 at varied positions, each
/// attending causally over the cached KV. `positions` also indexes the mask.
const POSITIONS: [i32; 4] = [3, 8, 14, (N_KV - 1) as i32];

struct Fires {
    qt: Tensor,
    indptr: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    // mask-disabled tables (prefill / prefill_lse)
    open_mask: Tensor,
    open_enabled: Tensor,
    // mask-enabled tables (masked): one keep byte per (row, key), stride N_KV
    keep_mask: Tensor,
    keep_enabled: Tensor,
    sm_scale: f32,
    rows: usize,
    head_dim: usize,
}

fn build_fires(rig: &mut Rig, head_dim: usize) -> Fires {
    let rows = POSITIONS.len();
    let hpos = rig.i32s(&POSITIONS);
    let hreq = rig.i32s(&[0i32; 4]);
    let hindptr = rig.u32s(&[0u32, rows as u32]); // one request owning all rows

    let open_mask = rig.u8s(&[0u8; 4]); // never read while disabled
    let open_enabled = rig.u8s(&[0u8; 4]); // 0 → full causal, mask off

    // A dense all-keep mask, stride N_KV: causal is still enforced by the kernel
    // (kp > q_pos is rejected independently), so an all-ones mask exercises the
    // masked kernel's mask-read path while leaving the attended set = causal.
    let keep_mask = rig.u8s(&[1u8; N_KV * 4]);
    let keep_enabled = rig.u8s(&[1u8; 4]); // 1 → apply the mask

    // seeded queries, bf16-rounded
    let q_src: Vec<f32> = (0..(rows * Q_HEADS * head_dim) as u64)
        .map(|i| 0.5 * gaussian(i ^ 0x0071))
        .collect();
    let hq = rig.bf16(&q_src);

    Fires {
        qt: Tensor::new(hq, rows as u32, (Q_HEADS * head_dim) as u32, Dtype::Bf16),
        indptr: Tensor::new(hindptr, 2, 1, Dtype::U32),
        positions: Tensor::new(hpos, rows as u32, 1, Dtype::I32),
        request_of_token: Tensor::new(hreq, rows as u32, 1, Dtype::I32),
        open_mask: Tensor::new(open_mask, rows as u32, 1, Dtype::U8),
        open_enabled: Tensor::new(open_enabled, rows as u32, 1, Dtype::U8),
        keep_mask: Tensor::new(keep_mask, rows as u32, N_KV as u32, Dtype::U8),
        keep_enabled: Tensor::new(keep_enabled, rows as u32, 1, Dtype::U8),
        sm_scale: (head_dim as f32).sqrt().recip(),
        rows,
        head_dim,
    }
}

/// Fire one tiled read `op` over `pool`; return (output plane f32, optional lse).
fn read_over(rig: &Rig, pool: &KvPool, op: Op, f: &Fires) -> (Vec<f32>, Option<Vec<f32>>) {
    let head_dim = f.head_dim;
    let out_elems = f.rows * Q_HEADS * head_dim;
    let (ho, obytes) = {
        let b = Buffer::zeroed(&rig.device, (out_elems * 2) as u64).expect("out");
        let h = rig.handles.bind(&b, 0, b.bytes()).expect("out handle");
        Box::leak(Box::new(b)); // outlive the fire (handles index it)
        (h, (out_elems * 2) as u64)
    };
    let ot = Tensor::new(ho, f.rows as u32, (Q_HEADS * head_dim) as u32, Dtype::Bf16);

    let lse_elems = f.rows * Q_HEADS;
    let (hl, lbytes) = {
        let b = Buffer::zeroed(&rig.device, (lse_elems * 4) as u64).expect("lse");
        let h = rig.handles.bind(&b, 0, b.bytes()).expect("lse handle");
        Box::leak(Box::new(b));
        (h, (lse_elems * 4) as u64)
    };
    let lt = Tensor::new(hl, f.rows as u32, Q_HEADS as u32, Dtype::F32);

    let (plan_open, plan_keep) = (
        PrefillPlan {
            positions: f.positions,
            request_of_token: f.request_of_token,
            mask: f.open_mask,
            mask_enabled: f.open_enabled,
            mask_stride: 1,
        },
        PrefillPlan {
            positions: f.positions,
            request_of_token: f.request_of_token,
            mask: f.keep_mask,
            mask_enabled: f.keep_enabled,
            mask_stride: N_KV as u32,
        },
    );
    let qrag = RaggedTensor {
        data: f.qt,
        indptr: f.indptr,
    };

    rig.fire(|s| match op {
        Op::Prefill => {
            attn::prefill(
                s,
                qrag,
                &plan_open,
                pool,
                None,
                head_dim as u32,
                KV_HEADS as u32,
                f.sm_scale,
                ot,
            )
            .expect("prefill launch");
        }
        Op::Masked => {
            attn::masked(
                s,
                qrag,
                &plan_keep,
                f.keep_mask,
                pool,
                None,
                head_dim as u32,
                true,
                f.sm_scale,
                ot,
            )
            .expect("masked launch");
        }
        Op::PrefillLse => {
            attn::prefill_lse(
                s,
                qrag,
                &plan_open,
                pool,
                None,
                head_dim as u32,
                KV_HEADS as u32,
                f.sm_scale,
                ot,
                lt,
            )
            .expect("prefill_lse launch");
        }
    });

    let out = bf16_floats(&rig.handles.read(ho, obytes).expect("read out"));
    let lse = match op {
        Op::PrefillLse => Some(f32_floats(&rig.handles.read(hl, lbytes).expect("read lse"))),
        _ => None,
    };
    (out, lse)
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

/// The per-(case, op) measurement: pack the seeded K/V, read over the packed
/// cache, over a bf16 cache holding the K/V UNPACKED from the packed cache (the
/// isolate reference), and over a bf16 cache holding the ORIGINAL K/V (the
/// quant-error reference). Returns ((iso max, iso mean), (quant max, quant mean),
/// optional lse iso max) for the given op.
fn run_case(
    rig: &mut Rig,
    label: &str,
    op: Op,
    fires: &Fires,
    k_src: &[f32],
    v_src: &[f32],
) -> ((f64, f64), (f64, f64), Option<f64>) {
    let head_dim = fires.head_dim;
    let width = (KV_HEADS * head_dim) as u32;
    let per = packed_bytes(head_dim);
    // ---- seed the write tables ----------------------------------------------
    let hk = rig.bf16(k_src);
    let hv = rig.bf16(v_src);
    let w_page = rig.u32s(&[0u32; N_KV]);
    let w_off = rig.u32s(&(0..N_KV as u32).collect::<Vec<_>>());
    let kt = Tensor::new(hk, N_KV as u32, width, Dtype::Bf16);
    let vt = Tensor::new(hv, N_KV as u32, width, Dtype::Bf16);
    let wpt = Tensor::new(w_page, N_KV as u32, 1, Dtype::U32);
    let wot = Tensor::new(w_off, N_KV as u32, 1, Dtype::U32);

    // ---- the packed cache + the original bf16 cache (same seeded K/V) --------
    let (packed, keys_h, values_h, kbytes) = packed_pool(rig, head_dim);
    write_kv(rig, &packed, kt, vt, wpt, wot);
    let orig_bf16 = bf16_pool(rig, head_dim);
    write_kv(rig, &orig_bf16, kt, vt, wpt, wot);

    // ---- reference cache: the K/V UNPACKED from the packed cache -------------
    // Read the packed bytes back, dequantize each (slot, head) 256-block to fp32,
    // lay it out as [N_KV, KV_HEADS*HEAD_DIM], and append into a fresh bf16 pool.
    // Reading over this pool is what a correct packed read must reproduce (up to
    // bf16 storage rounding of the dequantized values).
    let keys_raw = rig.handles.read(keys_h, kbytes).expect("read packed keys");
    let values_raw = rig
        .handles
        .read(values_h, kbytes)
        .expect("read packed values");
    let mut k_deq = vec![0.0f32; N_KV * KV_HEADS * head_dim];
    let mut v_deq = vec![0.0f32; N_KV * KV_HEADS * head_dim];
    for i in 0..N_KV {
        for h in 0..KV_HEADS {
            let slot = i; // page 0, offset i
            let base = (slot * KV_HEADS + h) * per;
            let kb = unpack_block(&keys_raw[base..base + per], head_dim);
            let vb = unpack_block(&values_raw[base..base + per], head_dim);
            let row = i * KV_HEADS * head_dim + h * head_dim;
            k_deq[row..row + head_dim].copy_from_slice(&kb);
            v_deq[row..row + head_dim].copy_from_slice(&vb);
        }
    }
    let hkd = rig.bf16(&k_deq);
    let hvd = rig.bf16(&v_deq);
    let ktd = Tensor::new(hkd, N_KV as u32, width, Dtype::Bf16);
    let vtd = Tensor::new(hvd, N_KV as u32, width, Dtype::Bf16);
    let unpacked_bf16 = bf16_pool(rig, head_dim);
    write_kv(rig, &unpacked_bf16, ktd, vtd, wpt, wot);

    // ---- the three reads ----------------------------------------------------
    let (out_packed, lse_packed) = read_over(rig, &packed, op, fires);
    let (out_unpacked, lse_unpacked) = read_over(rig, &unpacked_bf16, op, fires);
    let (out_orig, _) = read_over(rig, &orig_bf16, op, fires);

    let iso = deviation(&out_packed, &out_unpacked);
    let quant = deviation(&out_packed, &out_orig);
    let lse_iso = match (lse_packed, lse_unpacked) {
        (Some(a), Some(b)) => Some(deviation(&a, &b).0),
        _ => None,
    };

    eprintln!(
        "[{label:>10} d={head_dim:>3} {:>11} isolate] packed vs bf16-on-unpacked-from-packed: max={:.3e} mean={:.3e}",
        op.label(),
        iso.0,
        iso.1
    );
    eprintln!(
        "[{label:>10} d={head_dim:>3} {:>11} quant  ] packed vs bf16-on-original K/V:         max={:.3e} mean={:.3e}",
        op.label(),
        quant.0,
        quant.1
    );
    if let Some(m) = lse_iso {
        eprintln!(
            "[{label:>10} d={head_dim:>3} {:>11} lse-iso] packed lse vs bf16-on-unpacked lse:  max={m:.3e}",
            op.label()
        );
    }
    (iso, quant, lse_iso)
}

#[test]
fn the_packed_kv_tiled_reads() {
    eprintln!(
        "=== C2c-2: dequantizing tiled read (prefill / masked / prefill_lse) for packed 4-bit KV ==="
    );
    let Some(mut rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    eprintln!("device: {}", rig.device.name());

    // The codec block IS the head_dim; prove the packed tiled read at 256 AND 128.
    for &head_dim in &HEAD_DIMS {
        eprintln!("--- head_dim {head_dim} (codec block {head_dim}) ---");
        let fires = build_fires(&mut rig, head_dim);

        for op in [Op::Prefill, Op::Masked, Op::PrefillLse] {
            // ---- CASE 1: realistic random K/V (off-grid) ------------------------
            // Off-grid Gaussians with modest spikes: quantization is lossy (the FYI
            // quant number is meaty) and the isolate deviation is bounded by bf16
            // STORAGE rounding of the reconstructed values — the output is a
            // softmax-convex combination of V, so it cannot deviate by more than the
            // bf16 ulp of the biggest V it mixes. A wrong offset (q vs q+8), swapped
            // nibble parity, or a mis-indexed staging element corrupts the
            // reconstructed K/V across every block and drives the mean up by orders of
            // magnitude. Assert the mean is tight AND the whole isolate error sits far
            // below the real 4-bit quant error (the read reconstructs the INTENDED
            // values, it is not merely landing inside quantization noise).
            let ((r_iso_max, r_iso_mean), (r_q_max, _), _) = run_case(
                &mut rig,
                "random",
                op,
                &fires,
                &plane(N_KV, KV_HEADS * head_dim, head_dim, 0x0C2C),
                &plane(N_KV, KV_HEADS * head_dim, head_dim, 0x0DEC),
            );
            assert!(
                r_iso_mean < 2.0e-3,
                "{} d={head_dim}: the packed read parts from the bf16 read on the SAME (unpacked) K/V \
             by mean {r_iso_mean:.3e} — a systematic dequant/offset/parity bug, not bf16 rounding",
                op.label()
            );
            assert!(
                r_iso_max < 3.0e-2,
                "{} d={head_dim}: the packed read's worst element parts by {r_iso_max:.3e}, above the \
             bf16 storage-rounding ceiling — surface it as a read bug, do not loosen",
                op.label()
            );
            assert!(
                r_iso_max * 6.0 < r_q_max,
                "{} d={head_dim}: the isolate deviation {r_iso_max:.3e} is not clearly below the 4-bit \
             quant error {r_q_max:.3e}; the read must reconstruct the intended K/V, not merely \
             land within quantization noise",
                op.label()
            );
            assert!(
                r_q_max.is_finite() && r_q_max > 0.0,
                "{} d={head_dim}: the 4-bit quant error should be finite and nonzero, got {r_q_max:.3e}",
                op.label()
            );

            // ---- CASE 2: bf16-EXACT K/V (on-grid) — the unambiguous exactness proof
            // Every seed is q*S for q in [-7,7], S = 2^-2, with ±7*S planted per block.
            // Quantization is LOSSLESS and every dequant value is bf16-exact, so the
            // packed staging load and the bf16 reference stage BIT-IDENTICAL K/V. The
            // two tiled paths run the SAME fp32 body over the SAME inputs, so they must
            // agree to fp32-accumulation noise (expected ~0). A TIGHT max here is a
            // direct proof the dequant (offset, parity, scale, byte-index) is exact.
            let ((e_iso_max, e_iso_mean), (_e_q_max, _), e_lse_iso) = run_case(
                &mut rig,
                "bf16-exact",
                op,
                &fires,
                &exact_plane(N_KV, KV_HEADS * head_dim, head_dim, 0x0E7A),
                &exact_plane(N_KV, KV_HEADS * head_dim, head_dim, 0x0E7B),
            );
            assert!(
                e_iso_max < 1.0e-5,
                "{} d={head_dim}: the packed read parts from the bf16 read on BIT-IDENTICAL (on-grid) \
             K/V by max {e_iso_max:.3e} — the dequant is NOT exact (a real \
             offset/parity/scale/index bug), not a tolerance to loosen",
                op.label()
            );
            if let Some(m) = e_lse_iso {
                assert!(
                    m < 1.0e-5,
                    "{} d={head_dim}: the packed lse parts from the bf16 lse on BIT-IDENTICAL K/V by \
                 max {m:.3e} — the lse variant's dequant is NOT exact",
                    op.label()
                );
            }
            eprintln!(
                "[{:>11} d={head_dim:>3}] bf16-exact isolate max {e_iso_max:.3e} mean {e_iso_mean:.3e} — dequant is exact on-grid",
                op.label()
            );
        }
    }

    eprintln!(
        "=== C2c-2 done: the packed 4-bit KV tiled read dequantizes + attends correctly (prefill / masked / prefill_lse) at d=128 & 256 ==="
    );
}
