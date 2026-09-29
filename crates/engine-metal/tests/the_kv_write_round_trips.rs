#![cfg(target_vendor = "apple")]

//! C2b — the QUANTIZING paged KV write path, end to end.
//!
//! Where C2a drove the codec math in isolation, this drives the real
//! `attention.kv_append` op onto a paged KV pool whose dtype is the new packed
//! `KvU4` (spelling `g256_u4_f16_n`): the incoming bf16 K/V heads are quantized
//! in place into the LOCKED v1 KV format (4-bit symmetric absmax, block = 256,
//! one inline fp16 scale, 130 B/block, offset-binary nibbles). Two things are
//! proven:
//!
//!   (A) ALLOCATION: the store's demand accounting sizes a packed KV cache from
//!       the format's `row_bytes` (130 B per 256-block), not `width * element`.
//!       The packed cache is ~130/512 of the bf16 cache — the memory win is
//!       real and comes straight out of the allocation layer.
//!
//!   (B) WRITE: the bytes the kernel leaves in the pool decode (host-side) to a
//!       self-consistent 4-bit reconstruction. The scale is rounded to fp16
//!       FIRST and the codes are quantized against that SAME fp16 scale, so
//!       re-quantizing the bf16 source against the STORED scale reproduces the
//!       written codes bit-for-bit — the production self-consistency property.
//!       (The sdpa READ path is C2c and is not exercised here.)

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use engine_metal::store::kv::Paging;
use engine_metal::store::pool_demand;
use kernels_metal::attn;
use kernels_metal::{KvPool, Tensor};
use model_ir::{CacheRow, Dtype, Platform, Trace};

const BLOCK: usize = 256;
const PACKED: usize = 130; // 128 nibble bytes + one fp16 scale

// --- deterministic outlier-heavy KV data (same mixer as the C2a test) --------

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

/// A `rows x (heads*head_dim)` plane of small Gaussians with a few big spikes
/// planted per 256-block — the outlier shape a plain absmax quantizer strains on.
fn outlier_plane(rows: usize, width: usize, salt: u64) -> Vec<f32> {
    const SPIKES: usize = 3;
    let mut v: Vec<f32> = (0..(rows * width) as u64)
        .map(|i| gaussian(i ^ salt))
        .collect();
    for blk in 0..(rows * width) / BLOCK {
        let base = blk * BLOCK;
        for k in 0..SPIKES {
            let key =
                (blk as u64).wrapping_mul(0x100_0193) ^ (k as u64).wrapping_mul(0x9E37) ^ salt;
            let pos = (noise(key) as usize) % BLOCK;
            let mag = 20.0 + 20.0 * (unit01(key ^ 0xBEEF) as f32);
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

// --- bf16 / fp16 bit helpers -------------------------------------------------

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The value the kernel actually sees for a seeded f32: rounded to bf16.
fn bf16_view(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&f| f32_to_bf16_bits(f).to_le_bytes())
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

/// Pull the integer codes [-7, 7] and the fp16 scale out of one 130-byte block.
fn parse_block(bytes: &[u8]) -> ([i32; BLOCK], f32) {
    let mut codes = [0i32; BLOCK];
    for byte_i in 0..128 {
        let byte = bytes[byte_i];
        codes[2 * byte_i] = i32::from(byte & 0xf) - 8;
        codes[2 * byte_i + 1] = i32::from(byte >> 4) - 8;
    }
    let scale_bits = u16::from(bytes[128]) | (u16::from(bytes[129]) << 8);
    (codes, f16_bits_to_f32(scale_bits))
}

// --- a hand-built packed KV pool + u32/bf16 buffer helpers -------------------

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    // Buffers must outlive the handles that index them.
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
        let bytes = bf16_bytes(data);
        let mut b = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        b.write(0, &bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.push(b);
        h
    }

    fn u32s(&mut self, data: &[u32]) -> u32 {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut b = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a table");
        b.write(0, &bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.push(b);
        h
    }

    /// A zeroed packed plane of `cells` slots, `heads` blocks each — the shape
    /// the store would reserve for a `KvU4` cache.
    fn packed_plane(&mut self, cells: usize, heads: usize) -> (u32, u64) {
        let bytes = (cells * heads * PACKED) as u64;
        let b = Buffer::zeroed(&self.device, bytes).expect("packed plane");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.push(b);
        (h, bytes)
    }
}

// --- the test ----------------------------------------------------------------

const HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const TOKENS: usize = 6;
const PAGE_SIZE: u32 = 8;
const PAGES: u64 = 1;

#[test]
fn the_kv_write_round_trips() {
    // ---- (A) allocation: the packed cache is sized from row_bytes -----------
    eprintln!("=== C2b: packed 4-bit KV write (dtype g256_u4_f16_n) ===");
    assert_eq!(
        Dtype::KvU4.row_bytes(BLOCK as u32),
        Some(PACKED as u64),
        "the packed KV dtype must size a 256-block at 130 bytes"
    );

    let paging = Paging::of(PAGE_SIZE, PAGE_SIZE, 0, PAGES).expect("paging");
    let cells = paging.pages() * u64::from(PAGE_SIZE);
    let width = (HEADS * HEAD_DIM) as u64;
    let kv_trace = |dtype: Dtype| Trace {
        name: String::from("kv_size_probe"),
        platform: Platform::Metal,
        params: Vec::new(),
        // a key half and a value half, each `width` elements wide
        caches: vec![CacheRow::Kv {
            name: String::from("kv"),
            planes: vec![width, width],
            dtype,
            space: 0,
        }],
        values: Vec::new(),
        nodes: Vec::new(),
        seams: Vec::new(),
        drafter: None,
    };
    let bf16_bytes_total = pool_demand(&kv_trace(Dtype::Bf16), paging).expect("bf16 demand");
    let packed_bytes_total = pool_demand(&kv_trace(Dtype::KvU4), paging).expect("packed demand");

    // bf16: two planes of `cells * width * 2`. packed: two planes of
    // `cells * (width/256) * 130`.
    let want_bf16 = cells * width * 2 /*bytes*/ * 2 /*planes*/;
    let blocks_per_plane = width / BLOCK as u64;
    let want_packed = cells * blocks_per_plane * PACKED as u64 * 2 /*planes*/;
    assert_eq!(bf16_bytes_total, want_bf16, "bf16 KV cache sizing");
    assert_eq!(packed_bytes_total, want_packed, "packed KV cache sizing");
    let ratio = packed_bytes_total as f64 / bf16_bytes_total as f64;
    eprintln!(
        "[alloc ] bf16 cache = {bf16_bytes_total} B, packed cache = {packed_bytes_total} B, \
         ratio = {ratio:.4} (expected {:.4} = 130/512)",
        130.0 / 512.0
    );
    assert!(
        (ratio - 130.0 / 512.0).abs() < 1e-9,
        "packed cache must be 130/512 of bf16, got {ratio}"
    );

    // ---- (B) the quantizing write path --------------------------------------
    let Some(mut rig) = Rig::open() else {
        eprintln!("not asked: no Metal device — allocation sizing already checked");
        return;
    };

    // Seed bf16 K and V planes: TOKENS rows of HEADS*256 elements.
    let k_data = outlier_plane(TOKENS, HEADS * HEAD_DIM, 0x0C2B);
    let v_data = outlier_plane(TOKENS, HEADS * HEAD_DIM, 0x0F00);

    let hk = rig.bf16(&k_data);
    let hv = rig.bf16(&v_data);
    let (keys_h, keys_bytes) = rig.packed_plane(cells as usize, HEADS);
    let (values_h, _values_bytes) = rig.packed_plane(cells as usize, HEADS);
    // page tables the write path does not read, but the KvPool type carries.
    let ppi = rig.u32s(&[0u32]);
    let ppp = rig.u32s(&[0u32, 1]);
    // token i lands in page 0 at in-page offset i.
    let w_page = rig.u32s(&[0u32; TOKENS]);
    let w_off = rig.u32s(&(0..TOKENS as u32).collect::<Vec<_>>());

    let pool = KvPool {
        keys: Tensor::new(keys_h, cells as u32, width as u32, Dtype::KvU4),
        values: Tensor::new(values_h, cells as u32, width as u32, Dtype::KvU4),
        page_indices: Tensor::new(ppi, 1, 1, Dtype::U32),
        page_indptr: Tensor::new(ppp, 2, 1, Dtype::U32),
        page_size: PAGE_SIZE as i32,
        seq_stride: width,
        head_stride: HEAD_DIM as u64,
    };
    let kt = Tensor::new(hk, TOKENS as u32, width as u32, Dtype::Bf16);
    let vt = Tensor::new(hv, TOKENS as u32, width as u32, Dtype::Bf16);
    let wpt = Tensor::new(w_page, TOKENS as u32, 1, Dtype::U32);
    let wot = Tensor::new(w_off, TOKENS as u32, 1, Dtype::U32);

    {
        let frame = rig.device.frame().expect("a frame");
        let sink = Sink::new(&rig.device, &frame, &rig.pipelines, &rig.handles);
        attn::kv_append(&sink, kt, vt, &pool, wpt, wot).expect("kv_append launch");
        frame.commit().expect("commit");
    }

    let keys_raw = rig.handles.read(keys_h, keys_bytes).expect("read keys");
    let values_raw = rig.handles.read(values_h, keys_bytes).expect("read values");

    // Host-side decode + self-consistency check, per (token, head) 256-block.
    let mut worst_scale_rel = 0.0f64;
    let mut sse = 0.0f64;
    let mut n = 0usize;
    let mut worst_abs = 0.0f64;
    for (label, src, raw) in [("K", &k_data, &keys_raw), ("V", &v_data, &values_raw)] {
        for i in 0..TOKENS {
            for h in 0..HEADS {
                let slot = i; // page 0, offset i
                let base = (slot * HEADS + h) * PACKED;
                let (codes, scale) = parse_block(&raw[base..base + PACKED]);

                // the block of bf16-viewed source values
                let mut absmax = 0.0f32;
                let mut block = [0.0f32; BLOCK];
                for d in 0..BLOCK {
                    let x = bf16_view(src[i * HEADS * HEAD_DIM + h * HEAD_DIM + d]);
                    block[d] = x;
                    absmax = absmax.max(x.abs());
                }

                // (i) the stored fp16 scale is absmax/7 to within an fp16 ulp.
                let want = absmax / 7.0;
                assert!(absmax > 0.0, "seeded blocks are never all-zero");
                worst_scale_rel =
                    worst_scale_rel.max(f64::from((scale - want).abs()) / f64::from(want));

                // (ii) SELF-CONSISTENCY: re-quantizing the source against the
                //      STORED fp16 scale reproduces the written codes exactly.
                //      (A non-self-consistent encoder — dividing by the full
                //      precision scale but storing the fp16 one — would disagree
                //      on boundary elements.)
                for d in 0..BLOCK {
                    let q = (block[d] / scale).round().clamp(-7.0, 7.0) as i32;
                    assert_eq!(
                        q, codes[d],
                        "{label} token {i} head {h} elem {d}: written code {} is not the \
                         self-consistent quantization {q} of the source against the stored scale",
                        codes[d]
                    );
                    assert!((-7..=7).contains(&codes[d]), "code out of 4-bit range");
                    let recon = codes[d] as f32 * scale;
                    let e = f64::from(block[d]) - f64::from(recon);
                    sse += e * e;
                    worst_abs = worst_abs.max(e.abs());
                    n += 1;
                }
            }
        }
    }
    let mse = sse / n as f64;
    eprintln!(
        "[write ] {n} elems over {TOKENS} tokens x {HEADS} heads (K+V) | codes self-consistent | \
         fp16 scale rel-dev {worst_scale_rel:.2e} | recon vs bf16 src: max={worst_abs:.4} MSE={mse:.5e}"
    );
    assert!(
        worst_scale_rel < 1.0e-3,
        "fp16 scale rel-dev {worst_scale_rel:.2e} exceeds the fp16 half-ulp bound"
    );
    assert!(
        mse.is_finite() && mse > 0.0,
        "the round-trip MSE should be a finite positive 4-bit error, got {mse:.3e}"
    );
    eprintln!("=== C2b done: the packed write is self-consistent and the cache is 130/512 ===");
}
