#![cfg(feature = "cuda")]

//! `quantize_fp8b128` on a bf16 plane dequantized from an fp8b128 one lands
//! codes of exactly its values: each block's exponent the least that holds
//! its amax, the original's or below it (a block whose codes all sit at or
//! under 224 drops one, under 112 two), each code the original's times the
//! power of two between. On synthetic blocks of each kind, and (`--ignored`)
//! on every block of DeepSeek-V4-Flash's `wo_a` as the serving artifact holds
//! it, against the checkpoint's own fp8 codes and exponents. And
//! `dequantize_fp8b128` lands the bf16 back, bit for bit.

use crate::common::Gpu;
use dtype::Dtype;
use kernels_cuda::linear::moe::{dequantize_fp8b128, quantize_fp8b128};
use kernels_cuda::linear::tc5;
use kernels_cuda::tensor::Tensor;

/// An e4m3 code's value.
fn e4m3(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f64::from(b & 0x7);
    sign * if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// The byte of scale `(row, col)` in tcgen05's atom layout of `cols` columns.
fn atom_offset(row: usize, col: usize, cols: usize) -> usize {
    (row / 128 * cols.div_ceil(4) + col / 4) * 512 + (row % 32) * 16 + (row / 32 % 4) * 4 + col % 4
}

/// A bf16's value.
fn bf16(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// The bf16 of an fp8b128 value, which it holds exactly.
fn bf16_of(code: u8, exponent: u8) -> u16 {
    let v = (e4m3(code) * 2f64.powi(i32::from(exponent) - 127)) as f32;
    let bits = v.to_bits();
    assert_eq!(bits & 0xFFFF, 0, "an fp8b128 value is a bf16");
    (bits >> 16) as u16
}

/// The kernel's codes and atom-laid exponents for a bf16 plane.
fn requantize(gpu: &mut Gpu, plane: &[u16], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let src = gpu.up(plane);
    let codes = gpu.zeros(rows * k);
    let atom_bytes = tc5::atom_bytes(rows, k, 128);
    let atoms = gpu.zeros(atom_bytes);
    quantize_fp8b128(
        &gpu.ctx(),
        Tensor::new(src, rows as u32, k as u32, Dtype::Bf16),
        codes,
        atoms,
    )
    .expect("the requantize fires");
    gpu.sync();
    (gpu.down(codes, rows * k), gpu.down(atoms, atom_bytes))
}

/// Blocks whose requantized values differ from the original's, and how
/// many blocks' exponents dropped by each amount (index 0: kept).
struct Tally {
    differ: Vec<(usize, usize)>,
    drops: [usize; 4],
}

/// Each block of the requantized plane against the original codes
/// (`[rows, k]`) and exponents (`[rows / 128][k / 128]`), value for value.
fn tally(
    orig_codes: &[u8],
    orig_scales: &[u8],
    codes: &[u8],
    atoms: &[u8],
    rows: usize,
    k: usize,
) -> Tally {
    let blocks = k / 128;
    let mut t = Tally {
        differ: Vec::new(),
        drops: [0; 4],
    };
    for rb in 0..rows / 128 {
        for kb in 0..blocks {
            let s = orig_scales[rb * blocks + kb];
            let mut same = true;
            let mut e = None;
            for r in rb * 128..rb * 128 + 128 {
                let er = atoms[atom_offset(r, kb, blocks)];
                // Every row of a block holds the block's one exponent.
                same &= *e.get_or_insert(er) == er;
                for c in kb * 128..kb * 128 + 128 {
                    let at = r * k + c;
                    let want = e4m3(orig_codes[at]) * 2f64.powi(i32::from(s) - 127);
                    let got = e4m3(codes[at]) * 2f64.powi(i32::from(er) - 127);
                    same &= want == got;
                }
            }
            if !same {
                t.differ.push((rb, kb));
            }
            let drop = i32::from(s) - i32::from(e.unwrap_or(s));
            // A zero block's exponent is the least there is: not a drop.
            let zero = (rb * 128..rb * 128 + 128).all(|r| {
                orig_codes[r * k + kb * 128..r * k + kb * 128 + 128]
                    .iter()
                    .all(|b| b & 0x7F == 0)
            });
            if !zero {
                t.drops[usize::try_from(drop.clamp(0, 3)).unwrap_or(3)] += 1;
            }
        }
    }
    t
}

/// A finite e4m3 code from `x`, at most `max` in magnitude (`max` itself a
/// code).
fn code(x: u64, max: u8) -> u8 {
    let mut b = (x & 0xFF) as u8;
    if b & 0x7F == 0x7F {
        b &= 0xF7;
    }
    if b & 0x7F > max {
        b = (b & 0x80) | ((b & 0x7F) % (max + 1));
    }
    b
}

#[test]
fn the_fp8b128_requantize_lands_every_value() {
    let (rows, k) = (256usize, 768usize);
    let blocks = k / 128;
    // A block's kind: the largest code magnitude it may hold (0x7E: 448;
    // 0x6F: 240, 0x6E: 224, which drops one; 0x5E: 112, which drops two),
    // all zeros, or subnormals alone.
    let kinds: [Option<u8>; 6] = [
        Some(0x7E),
        Some(0x6F),
        Some(0x6E),
        Some(0x5E),
        None,
        Some(0x07),
    ];
    let mut state = 0x5eed_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 24
    };
    let mut scales = vec![0u8; (rows / 128) * blocks];
    for (i, s) in scales.iter_mut().enumerate() {
        *s = 116 + (i % 9) as u8;
    }
    let mut codes = vec![0u8; rows * k];
    for r in 0..rows {
        for c in 0..k {
            let kind = kinds[(r / 128 * blocks + c / 128) % kinds.len()];
            codes[r * k + c] = match kind {
                None => 0,
                Some(max) => code(next(), max),
            };
        }
        // Each block that may reach its bound reaches it once.
        if r % 128 == 5 {
            for kb in 0..blocks {
                if let Some(max) = kinds[(r / 128 * blocks + kb) % kinds.len()] {
                    codes[r * k + kb * 128 + 3] = max | 0x80;
                }
            }
        }
    }
    let plane: Vec<u16> = (0..rows * k)
        .map(|at| bf16_of(codes[at], scales[(at / k / 128) * blocks + at % k / 128]))
        .collect();
    let mut gpu = Gpu::open();
    let (got, atoms) = requantize(&mut gpu, &plane, rows, k);
    let t = tally(&codes, &scales, &got, &atoms, rows, k);
    assert!(
        t.differ.is_empty(),
        "blocks whose values moved: {:?}",
        t.differ
    );
    assert!(
        t.drops[1] > 0,
        "a block that drops one exponent: {:?}",
        t.drops
    );
    assert!(t.drops[2] > 0, "a block that drops two: {:?}", t.drops);
    assert!(
        t.drops[0] > 0,
        "a block that keeps its exponent: {:?}",
        t.drops
    );

    // And back: `dequantize_fp8b128` lands the bf16 it was given, bit for bit.
    let codes_at = gpu.up(&got);
    let atoms_at = gpu.up(&atoms);
    let back_at = gpu.zeros(rows * k * 2);
    dequantize_fp8b128(
        &gpu.ctx(),
        codes_at,
        atoms_at,
        back_at,
        rows as u32,
        k as u32,
    )
    .expect("the dequantize fires");
    gpu.sync();
    let back: Vec<u16> = gpu.down(back_at, rows * k);
    assert!(
        back == plane,
        "the round trip lands the bf16 plane bit for bit"
    );
}

/// DeepSeek-V4-Flash's snapshot and its fp8b128a8 serving artifact (or
/// `PIE_DSV4_SNAPSHOT` and `PIE_DSV4_ARTIFACT`).
const SNAPSHOT: &str = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/\
                        snapshots/60d8d70770c6776ff598c94bb586a859a38244f1";
const ARTIFACT: &str = "/root/pie-homes/bench/artifacts/60d8d70770c6776ff598c94bb586a859a38244f1/\
                        60d8d70770c6776ff598c94bb586a859a38244f1.dsv4-flash-fp8b128a8-mxfp4a8-kv-bf16.cuda.zt";

#[test]
#[ignore = "reads DeepSeek-V4-Flash's checkpoint and serving artifact"]
fn every_dsv4_wo_a_block_requantizes_to_the_checkpoints_values() {
    let snapshot = std::env::var("PIE_DSV4_SNAPSHOT").unwrap_or_else(|_| SNAPSHOT.to_string());
    let artifact = std::env::var("PIE_DSV4_ARTIFACT").unwrap_or_else(|_| ARTIFACT.to_string());
    let shards: Vec<std::path::PathBuf> = std::fs::read_dir(&snapshot)
        .expect("the snapshot reads")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    let checkpoint = ztensor_compat::index_all(&shards).expect("the checkpoint indexes");
    let served = ztensor_compat::index(&artifact).expect("the artifact indexes");
    let (mut total, mut drops, mut layers) = (0usize, [0usize; 4], 0usize);
    let mut differ = Vec::new();
    let mut gpu = Gpu::open();
    for l in 0.. {
        let Some(w) = checkpoint.get(&format!("layers.{l}.attn.wo_a.weight")) else {
            break;
        };
        let s = checkpoint
            .tensor(&format!("layers.{l}.attn.wo_a.scale"))
            .expect("each wo_a has its exponents");
        let name = format!("layer.{l}.o_down");
        let plane = served.tensor(&name).unwrap_or_else(|_| {
            let near: Vec<&str> = served
                .names()
                .filter(|n| n.contains("o_down"))
                .take(4)
                .collect();
            panic!("the artifact holds no {name:?} (near: {near:?})")
        });
        let (rows, k) = (w.shape()[0] as usize, w.shape()[1] as usize);
        assert_eq!(plane.shape(), w.shape(), "{name} is wo_a's shape");
        let codes = w.bytes().expect("wo_a reads").into_owned();
        let scales = s.bytes().expect("its exponents read").into_owned();
        assert_eq!(
            scales.len(),
            (rows / 128) * (k / 128),
            "one exponent a 128x128 block"
        );
        let bits: Vec<u16> = plane
            .bytes()
            .expect("o_down reads")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes(*p))
            .collect();
        // The artifact's bf16 is the checkpoint's fp8b128, value for value.
        let blocks = k / 128;
        for (at, &b) in bits.iter().enumerate() {
            let want = e4m3(codes[at])
                * 2f64.powi(i32::from(scales[(at / k / 128) * blocks + at % k / 128]) - 127);
            assert_eq!(
                bf16(b),
                want,
                "{name}[{at}]: the artifact's bf16 is the checkpoint's value"
            );
        }
        let (got, atoms) = requantize(&mut gpu, &bits, rows, k);
        let t = tally(&codes, &scales, &got, &atoms, rows, k);
        total += (rows / 128) * blocks;
        for (d, n) in t.drops.iter().enumerate() {
            drops[d] += n;
        }
        differ.extend(t.differ.iter().map(|&(rb, kb)| (l, rb, kb)));
        layers += 1;
    }
    println!(
        "{layers} layers, {total} blocks: {} differ; exponents kept {}, one lower {}, two {}, \
         three or more {}",
        differ.len(),
        drops[0],
        drops[1],
        drops[2],
        drops[3]
    );
    assert!(layers > 0, "the checkpoint holds wo_a");
    assert!(
        differ.is_empty(),
        "blocks whose values moved: {:?}",
        &differ[..differ.len().min(16)]
    );
}
