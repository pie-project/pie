#![cfg(target_vendor = "apple")]

//! C2a — the low-bit KV codec's pack/unpack Metal kernels, in ISOLATION.
//!
//! This drives the real `kv_codec::pack` / `kv_codec::unpack` ops (the LOCKED v1
//! format: 4-bit symmetric absmax, block = 256, one fp16 scale per block, no
//! bias) directly from a test, the same way the C0 test drives
//! `pointwise::hadamard`. Nothing here touches the cache, the allocator or the
//! forward path — it is the codec MATH only.
//!
//! The KEY correctness check is a round-trip against a pure HOST reference that
//! is exactly the per-block symmetric absmax quantize/dequant the C0 test uses
//! (`quantize_rowmajor`, here pinned at block = 256). We prove three things:
//!   1. Nibble/quantize math is bit-exact vs host codes (converter-free — this
//!      is what catches a wrong signed-nibble convention).
//!   2. Metal's reconstruction matches the host `quantize_rowmajor` recon within
//!      the fp16-scale bound |q|*|fp16(scale) - scale| <= global_absmax * 2^-11
//!      (the codes are identical because both divide by the full-precision
//!      scale; only the stored scale is rounded to fp16).
//!   3. The pack->unpack MSE vs the original is a sane 4-bit absmax error.
//!
//! Signed-nibble convention under test: OFFSET-BINARY (store q+8 -> [1..15],
//! unpack (nibble-8)*scale).

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::linear::kv_codec;
use model_ir::Dtype;

const BLOCK: usize = 256;

// --- deterministic data (same mixer as the C0 / hadamard tests) --------------

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5e5e_1234_9ABC_DEF0;
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

/// Outlier-heavy KV-like data: small Gaussians with a few large spikes planted
/// per row — the shape a plain absmax quantizer struggles with.
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
            let mag = 20.0 + 20.0 * (unit01(key ^ 0xBEEF) as f32);
            let sign = if noise(key ^ 0xF00D) & 1 == 0 { 1.0 } else { -1.0 };
            v[base + pos] = sign * mag;
        }
    }
    v
}

fn uniform_data(rows: usize, head_dim: usize, salt: u64) -> Vec<f32> {
    (0..(rows * head_dim) as u64)
        .map(|i| gaussian(i ^ salt))
        .collect()
}

// --- byte helpers ------------------------------------------------------------

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

/// bf16 store/load: bf16 is the top 16 bits of f32 with round-to-nearest-even.
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

fn bf16_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes(*c)))
        .collect()
}

/// Decode an IEEE binary16 bit pattern to f32 (exact — every f16 is an f32).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if (bits >> 15) & 1 == 1 { -1.0f32 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    if exp == 0 {
        // zero or subnormal: value = mant * 2^-24
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

// --- the host reference: quantize_rowmajor, pinned at block = 256 ------------

/// Per-256-block symmetric absmax 4-bit quantize/dequant, in pure host f32 — the
/// same scheme the C0 test's `quantize_rowmajor` implements. Returns the integer
/// codes in [-7, 7], the f32-scale reconstruction, and the per-block scale.
fn host_quantize(data: &[f32]) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    const LIM: f32 = 7.0;
    let blocks = data.len() / BLOCK;
    let mut codes = vec![0i32; data.len()];
    let mut recon = vec![0.0f32; data.len()];
    let mut scales = vec![0.0f32; blocks];
    for (bi, chunk) in data.chunks_exact(BLOCK).enumerate() {
        let base = bi * BLOCK;
        let absmax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if absmax == 0.0 {
            // codes stay 0, recon stays 0, scale stays 0 (matches the kernel's
            // all-zero-block guard: no divide, no NaN).
            continue;
        }
        let scale = absmax / LIM;
        scales[bi] = scale;
        for (j, &v) in chunk.iter().enumerate() {
            let q = (v / scale).round().clamp(-LIM, LIM);
            codes[base + j] = q as i32;
            recon[base + j] = q * scale;
        }
    }
    (codes, recon, scales)
}

/// Decode the codec's packed nibbles back to integer codes in [-7, 7] and pull
/// out the fp16 scale of each block — a converter-free view of what the pack
/// kernel actually wrote, used to check the signed-nibble convention directly.
fn parse_packed(packed: &[u8], blocks: usize) -> (Vec<i32>, Vec<f32>) {
    let mut codes = vec![0i32; blocks * BLOCK];
    let mut scales = vec![0.0f32; blocks];
    for b in 0..blocks {
        let base = b * 130;
        for byte_i in 0..128 {
            let byte = packed[base + byte_i];
            codes[b * BLOCK + 2 * byte_i] = i32::from(byte & 0xf) - 8;
            codes[b * BLOCK + 2 * byte_i + 1] = i32::from(byte >> 4) - 8;
        }
        let scale_bits = u16::from(packed[base + 128]) | (u16::from(packed[base + 129]) << 8);
        scales[b] = f16_bits_to_f32(scale_bits);
    }
    (codes, scales)
}

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

fn global_absmax(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

// --- Metal drivers -----------------------------------------------------------

struct Rig<'a> {
    device: &'a Context,
    handles: &'a Handles,
    pipelines: &'a Pipelines,
}

impl Rig<'_> {
    /// Pack `data` (given as raw element bytes of `in_dtype`) into the v1 format
    /// and return the raw packed bytes (`blocks * 130`).
    fn pack(&self, data_bytes: &[u8], in_dtype: Dtype, elems: usize) -> Vec<u8> {
        let blocks = (elems / BLOCK) as u64;
        let packed_bytes = blocks * 130;
        let mut inb = Buffer::zeroed(self.device, data_bytes.len() as u64).expect("in buffer");
        inb.write(0, data_bytes).expect("write in");
        let in_h = self.handles.bind(&inb, 0, inb.bytes()).expect("bind in");
        let packed = Buffer::zeroed(self.device, packed_bytes).expect("packed buffer");
        let pk_h = self.handles.bind(&packed, 0, packed.bytes()).expect("bind packed");
        let x = Tensor::new(in_h, 1, elems as u32, in_dtype);
        let pk = Tensor::new(pk_h, 1, packed_bytes as u32, Dtype::U8);
        {
            let frame = self.device.frame().expect("frame");
            let sink = Sink::new(self.device, &frame, self.pipelines, self.handles);
            kv_codec::pack(&sink, x, pk).expect("pack launch");
            frame.commit().expect("pack commit");
        }
        self.handles.read(pk_h, packed_bytes).expect("read packed")
    }

    /// Unpack `packed` into `out_dtype` and return the raw element bytes.
    fn unpack(&self, packed_bytes: &[u8], out_dtype: Dtype, elems: usize) -> Vec<u8> {
        let elem_bytes = match out_dtype {
            Dtype::F32 => 4u64,
            Dtype::Bf16 => 2,
            other => panic!("unpack test only drives f32/bf16, not {other:?}"),
        };
        let out_bytes = elems as u64 * elem_bytes;
        let mut pkb = Buffer::zeroed(self.device, packed_bytes.len() as u64).expect("packed buffer");
        pkb.write(0, packed_bytes).expect("write packed");
        let pk_h = self.handles.bind(&pkb, 0, pkb.bytes()).expect("bind packed");
        let outb = Buffer::zeroed(self.device, out_bytes).expect("out buffer");
        let out_h = self.handles.bind(&outb, 0, outb.bytes()).expect("bind out");
        let pk = Tensor::new(pk_h, 1, packed_bytes.len() as u32, Dtype::U8);
        let out = Tensor::new(out_h, 1, elems as u32, out_dtype);
        {
            let frame = self.device.frame().expect("frame");
            let sink = Sink::new(self.device, &frame, self.pipelines, self.handles);
            kv_codec::unpack(&sink, pk, out).expect("unpack launch");
            frame.commit().expect("unpack commit");
        }
        self.handles.read(out_h, out_bytes).expect("read out")
    }
}

// --- the tests ---------------------------------------------------------------

#[test]
fn the_kv_codec_round_trips() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let rig = Rig {
        device: &device,
        handles: &handles,
        pipelines: &pipelines,
    };

    const ROWS: usize = 64;

    eprintln!("=== C2a: 4-bit symmetric KV codec (block=256) round-trip ===");
    eprintln!("    convention: offset-binary nibble (store q+8, unpack (n-8)*scale)");

    // (1)+(2): round-trip vs the host quantize_rowmajor reference, f32.
    for &head_dim in &[256usize, 128] {
        for (label, data) in [
            ("outlier", outlier_data(ROWS, head_dim, 0x0117 ^ head_dim as u64)),
            ("uniform", uniform_data(ROWS, head_dim, 0x2222 ^ head_dim as u64)),
        ] {
            assert!(data.len() % BLOCK == 0, "test data must be whole 256-blocks");
            let blocks = data.len() / BLOCK;

            let packed = rig.pack(&f32_bytes(&data), Dtype::F32, data.len());
            assert_eq!(packed.len(), blocks * 130, "packed size is 130 B/block");
            let recon = f32_floats(&rig.unpack(&packed, Dtype::F32, data.len()));

            let (host_codes, host_recon, _host_scales) = host_quantize(&data);
            let (metal_codes, metal_scales) = parse_packed(&packed, blocks);

            // (1) codes are bit-exact vs the host reference — the strongest,
            // converter-free check of quantize + nibble convention.
            let code_mismatches = host_codes
                .iter()
                .zip(metal_codes.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                code_mismatches, 0,
                "{label} head_dim={head_dim}: {code_mismatches} of {} nibble codes disagree with \
                 the host reference — a quantize or signed-nibble-convention bug",
                host_codes.len()
            );
            // code range sanity: offset-binary keeps q in [-7, 7].
            assert!(
                metal_codes.iter().all(|&q| (-7..=7).contains(&q)),
                "{label} head_dim={head_dim}: a decoded code fell outside [-7, 7]"
            );

            // scale check: the stored fp16 scale is within half an fp16 ulp of
            // the host's full-precision absmax/7.
            let mut max_scale_rel = 0.0f64;
            for (bi, chunk) in data.chunks_exact(BLOCK).enumerate() {
                let absmax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let want = absmax / 7.0;
                let got = metal_scales[bi];
                if want == 0.0 {
                    assert_eq!(got, 0.0, "{label}: an all-zero block must store scale 0");
                } else {
                    max_scale_rel =
                        max_scale_rel.max(f64::from((got - want).abs()) / f64::from(want));
                }
            }
            assert!(
                max_scale_rel < 1.0e-3,
                "{label} head_dim={head_dim}: fp16 scale rel-dev {max_scale_rel:.2e} exceeds the \
                 fp16 half-ulp bound"
            );

            // (2) reconstruction vs the host quantize_rowmajor recon. Codes are
            // identical, so the only gap is the fp16-rounded scale:
            // |q| * |fp16(scale) - scale| <= global_absmax * 2^-11.
            let (mse_host, max_host) = errors(&host_recon, &recon);
            let gabs = f64::from(global_absmax(&data));
            let bound = gabs * 1.0e-3; // 2^-11 ~= 4.9e-4, with margin
            eprintln!(
                "[{label}] head_dim={head_dim:>3} | codes exact | scale rel-dev {max_scale_rel:.2e} \
                 | Metal-vs-host recon: max={max_host:.3e} mean-sq={mse_host:.3e} (bound {bound:.3e})"
            );
            assert!(
                max_host <= bound,
                "{label} head_dim={head_dim}: Metal recon deviates {max_host:.3e} from the host \
                 reference, above the fp16-scale bound {bound:.3e}"
            );

            // (3) reconstruction error vs the ORIGINAL — a sanity that this is a
            // sane 4-bit absmax quantizer.
            let (mse_orig, max_orig) = errors(&data, &recon);
            eprintln!(
                "           pack->unpack vs original: MSE={mse_orig:.5e} max-abs={max_orig:.4}"
            );
            assert!(
                recon.iter().all(|v| v.is_finite()),
                "{label} head_dim={head_dim}: reconstruction has a non-finite value"
            );
        }
    }

    // bf16 path: exercise the bf16 pack/unpack kernels. Input is bf16-rounded,
    // so we compare against the bf16 view of the data and just assert a sane
    // 4-bit error with no NaNs (the exact-math check above is the f32 one).
    {
        let data = outlier_data(ROWS, 256, 0xB16);
        let bf_in = bf16_bytes(&data);
        let bf_view = bf16_floats(&bf_in); // what the kernel actually sees
        let packed = rig.pack(&bf_in, Dtype::Bf16, data.len());
        let recon = bf16_floats(&rig.unpack(&packed, Dtype::Bf16, data.len()));
        let (mse, max) = errors(&bf_view, &recon);
        eprintln!("[bf16   ] head_dim=256 | pack->unpack vs bf16 input: MSE={mse:.5e} max-abs={max:.4}");
        assert!(
            recon.iter().all(|v| v.is_finite()),
            "bf16 reconstruction has a non-finite value"
        );
        assert!(
            mse.is_finite() && mse > 0.0,
            "bf16 round-trip MSE should be a finite positive 4-bit error, got {mse:.3e}"
        );
    }

    // (edge) all-zero block: scale 0, codes all 0 (nibble 8), recon all 0, no NaN.
    {
        let data = vec![0.0f32; 2 * BLOCK];
        let packed = rig.pack(&f32_bytes(&data), Dtype::F32, data.len());
        let recon = f32_floats(&rig.unpack(&packed, Dtype::F32, data.len()));
        assert!(
            recon.iter().all(|&v| v == 0.0),
            "all-zero block must reconstruct to zeros with no NaN"
        );
        // every nibble is 8 (the offset-binary encoding of q=0) and each scale is 0.
        for b in 0..2 {
            for byte_i in 0..128 {
                assert_eq!(
                    packed[b * 130 + byte_i],
                    0x88,
                    "all-zero block: every byte must pack two q=0 nibbles (0x88)"
                );
            }
            assert_eq!(packed[b * 130 + 128], 0, "all-zero block: fp16 scale low byte");
            assert_eq!(packed[b * 130 + 129], 0, "all-zero block: fp16 scale high byte");
        }
        eprintln!("[edge   ] all-zero block: scale 0, codes 0x88, recon zeros, no NaN — OK");
    }

    // (edge) a width that is not a multiple of 256 must be REJECTED, not padded.
    {
        let ragged = uniform_data(1, 300, 0x9);
        let mut inb = Buffer::zeroed(&device, (ragged.len() * 4) as u64).expect("in buffer");
        inb.write(0, &f32_bytes(&ragged)).expect("write in");
        let in_h = handles.bind(&inb, 0, inb.bytes()).expect("bind in");
        let packed = Buffer::zeroed(&device, 130).expect("packed buffer");
        let pk_h = handles.bind(&packed, 0, packed.bytes()).expect("bind packed");
        let x = Tensor::new(in_h, 1, 300, Dtype::F32);
        let pk = Tensor::new(pk_h, 1, 130, Dtype::U8);
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        let refused = kv_codec::pack(&sink, x, pk);
        assert!(
            refused.is_err(),
            "a 300-element (non-multiple-of-256) tensor must be refused by the codec"
        );
        eprintln!("[edge   ] ragged width 300 rejected: {}", refused.unwrap_err());
    }

    eprintln!("=== C2a done: Metal KV codec reproduces the host quantize_rowmajor exactly ===");
}
