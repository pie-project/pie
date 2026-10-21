//! The quantized KV cache against a host reference: every scheme's append
//! (the codes and scales it lands, byte for byte, through a shuffled page
//! table and across two appends), the dequant into the bf16 shadow a
//! decode refreshes, and the decode's attention over that shadow.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, KV, e4m3_value, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, kv::KvScheme, plan};
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

// ------------------------------------------------------- host formats --

/// The nearest finite e4m3 code, a tie to the even code, saturating at 448.
fn e4m3_code(x: f32) -> u8 {
    let a = x.abs();
    let mut best = 0u8;
    for c in 1..=0x7eu8 {
        let (dc, db) = ((e4m3_value(c) - a).abs(), (e4m3_value(best) - a).abs());
        if dc < db || (dc == db && c % 2 == 0) {
            best = c;
        }
    }
    if x.is_sign_negative() {
        best | 0x80
    } else {
        best
    }
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1_code(x: f32) -> u8 {
    let a = x.abs();
    let mut best = 0;
    for (i, level) in E2M1.iter().enumerate() {
        if (a - level).abs() < (a - E2M1[best]).abs() {
            best = i;
        }
    }
    (if x < 0.0 { 8 } else { 0 }) | best as u8
}

fn e2m1_value(code: u8) -> f32 {
    let v = E2M1[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

// ------------------------------------------------------------- a pool --

struct Case {
    scheme: KvScheme,
    kv_heads: u32,
    head_dim: u32,
    page_size: u32,
    /// The scale of the keys' values: past 448 an fp8 code saturates.
    key_scale: f32,
}

const FP4_BLOCK: usize = 16;

/// The pool's geometry for requests of `kv_lens` tokens, pages handed out in
/// reverse so a wrong page walk shows.
struct Pages {
    total: usize,
    indices: Vec<i32>,
    indptr: Vec<i32>,
}

fn pages(kv_lens: &[u32], ps: usize) -> Pages {
    let per: Vec<usize> = kv_lens
        .iter()
        .map(|&n| (n as usize).div_ceil(ps).max(1))
        .collect();
    let total = per.iter().sum::<usize>() + 2;
    let mut indices = Vec::new();
    let mut indptr = vec![0];
    for &p in &per {
        for _ in 0..p {
            indices.push((total - 1 - indices.len()) as i32);
        }
        indptr.push(indices.len() as i32);
    }
    Pages {
        total,
        indices,
        indptr,
    }
}

/// The codes, scales and shadows of one pool, and the pool itself.
struct Quantized {
    pool: KvPool,
    k_codes: Tensor,
    v_codes: Tensor,
    k_scales: Tensor,
    v_scales: Tensor,
}

fn quantized(gpu: &Gpu, c: &Case, pg: &Pages) -> Quantized {
    let (h, d, ps) = (
        c.kv_heads as usize,
        c.head_dim as usize,
        c.page_size as usize,
    );
    let slots = pg.total * ps;
    let code_width = if c.scheme == KvScheme::Fp4Block {
        h * d.div_ceil(2)
    } else {
        h * d
    };
    let dtype = if c.scheme == KvScheme::Int8PerTokenHead {
        Dtype::I8
    } else {
        Dtype::U8
    };
    let codes = |seed: u8| {
        gpu.upload(
            &vec![seed; slots * code_width],
            slots as u32,
            code_width as u32,
            dtype,
        )
    };
    let scale_width = match c.scheme {
        KvScheme::Fp4Block => h * d.div_ceil(FP4_BLOCK),
        _ => h,
    };
    let scales = || gpu.zeros(slots as u32, scale_width as u32, Dtype::F32);
    let (k_codes, v_codes) = (codes(0x5a), codes(0x5a));
    let (k_scales, v_scales) = (scales(), scales());
    let pool = KvPool {
        keys: k_codes,
        values: v_codes,
        bf16_keys: gpu.zeros(slots as u32, (h * d) as u32, Dtype::Bf16),
        bf16_values: gpu.zeros(slots as u32, (h * d) as u32, Dtype::Bf16),
        key_scales: k_scales,
        value_scales: v_scales,
        page_indices: gpu.column(&pg.indices),
        page_indptr: gpu.column(&pg.indptr),
        page_size: ps as i32,
        seq_stride: (h * d) as i64,
        head_stride: d as i64,
        scheme_byte: c.scheme as i32,
        block_size: if c.scheme == KvScheme::Fp4Block {
            FP4_BLOCK as i32
        } else {
            0
        },
        pages_in_batch: pg.indices.len() as i32,
        live: [0, 0],
        ..KV
    };
    Quantized {
        pool,
        k_codes,
        v_codes,
        k_scales,
        v_scales,
    }
}

fn last_page_lens(kv_lens: &[u32], ps: u32) -> Vec<i32> {
    kv_lens
        .iter()
        .map(|&n| {
            let pages = n.div_ceil(ps).max(1);
            (n - (pages - 1) * ps) as i32
        })
        .collect()
}

/// One head's codes and scales as the device should land them.
fn quantize_head(scheme: KvScheme, x: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    match scheme {
        KvScheme::Fp8PerTensor => (x.iter().map(|&v| e4m3_code(v)).collect(), vec![]),
        KvScheme::Fp8PerTokenHead | KvScheme::Int8PerTokenHead => {
            let fp8 = scheme == KvScheme::Fp8PerTokenHead;
            let qmax = if fp8 { 448.0 } else { 127.0 };
            let scale = if amax > 0.0 { amax / qmax } else { 1.0 };
            let inv = 1.0 / scale;
            let codes = x
                .iter()
                .map(|&v| {
                    if fp8 {
                        e4m3_code(v * inv)
                    } else {
                        (v * inv).round_ties_even().clamp(-128.0, 127.0) as i8 as u8
                    }
                })
                .collect();
            (codes, vec![scale])
        }
        KvScheme::Fp4Block => {
            let mut codes = vec![0u8; x.len().div_ceil(2)];
            let mut scales = Vec::new();
            for (b, block) in x.chunks(FP4_BLOCK).enumerate() {
                let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = if amax > 0.0 { amax / 6.0 } else { 1.0 };
                let inv = 1.0 / scale;
                for (i, &v) in block.iter().enumerate() {
                    let j = b * FP4_BLOCK + i;
                    codes[j / 2] |= e2m1_code(v * inv) << (4 * (j % 2));
                }
                scales.push(scale);
            }
            (codes, scales)
        }
        KvScheme::Native => unreachable!(),
    }
}

/// One head's values back from its codes and scales.
fn dequantize_head(scheme: KvScheme, codes: &[u8], scales: &[f32], d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| match scheme {
            KvScheme::Fp8PerTensor => e4m3_value(codes[j]),
            KvScheme::Fp8PerTokenHead => e4m3_value(codes[j]) * scales[0],
            KvScheme::Int8PerTokenHead => f32::from(codes[j] as i8) * scales[0],
            KvScheme::Fp4Block => e2m1_value(codes[j / 2] >> (4 * (j % 2))) * scales[j / FP4_BLOCK],
            KvScheme::Native => unreachable!(),
        })
        .map(round_bf16)
        .collect()
}

/// Appends `fresh[r]` tokens to requests holding `prefix[r]`, in two fires
/// (the prefix, then the rest), and checks the codes and scales of every
/// token; then a decode of each request's last token reads the shadow its
/// dequant refreshes. Returns nothing; asserts.
fn run(gpu: &Gpu, c: &Case, prefix: &[u32], fresh: &[u32]) {
    let ctx = gpu.ctx();
    let (h, d, ps) = (
        c.kv_heads as usize,
        c.head_dim as usize,
        c.page_size as usize,
    );
    let row = h * d;
    let kv_lens: Vec<u32> = prefix.iter().zip(fresh).map(|(p, f)| p + f).collect();
    let pg = pages(&kv_lens, ps);
    let mut qz = quantized(gpu, c, &pg);

    // Host keys and values, [request][token][head][d].
    let tokens = |r: usize, seed: u64, scale: f32| -> Vec<f32> {
        values(kv_lens[r] as usize * row, seed + r as u64, scale)
            .into_iter()
            .map(round_bf16)
            .collect()
    };
    let keys: Vec<Vec<f32>> = (0..kv_lens.len())
        .map(|r| tokens(r, 21, c.key_scale))
        .collect();
    let vals: Vec<Vec<f32>> = (0..kv_lens.len()).map(|r| tokens(r, 61, 1.0)).collect();

    // Fire 1 appends each prefix, fire 2 the fresh rows after it.
    for (lens, from) in [
        (prefix.to_vec(), vec![0; prefix.len()]),
        (fresh.to_vec(), prefix.to_vec()),
    ] {
        let after: Vec<u32> = lens.iter().zip(&from).map(|(l, f)| l + f).collect();
        let mut indptr = vec![0i32];
        let (mut kp, mut vp) = (Vec::new(), Vec::new());
        for r in 0..lens.len() {
            indptr.push(indptr[r] + lens[r] as i32);
            let span = from[r] as usize * row..after[r] as usize * row;
            kp.extend_from_slice(&keys[r][span.clone()]);
            vp.extend_from_slice(&vals[r][span]);
        }
        let n = *indptr.last().unwrap() as u32;
        if n == 0 {
            continue;
        }
        // The page table names the pages the request will have; what the
        // append reads is how full the last of them is after it.
        let mut pool = qz.pool;
        pool.last_page_lens = gpu.column(&last_page_lens(&after, c.page_size));
        let mut page_indptr = vec![0i32];
        let mut page_indices = Vec::new();
        for (r, &a) in after.iter().enumerate() {
            let have = (a as usize).div_ceil(ps).max(1);
            let first = pg.indptr[r] as usize;
            page_indices.extend_from_slice(&pg.indices[first..first + have]);
            page_indptr.push(page_indices.len() as i32);
        }
        pool.page_indices = gpu.column(&page_indices);
        pool.page_indptr = gpu.column(&page_indptr);
        let k = RaggedTensor {
            data: gpu.tensor(&kp, n, row as u32, Dtype::Bf16),
            indptr: gpu.column(&indptr),
        };
        let v = gpu.tensor(&vp, n, row as u32, Dtype::Bf16);
        attn::kv_append(&ctx, k, v, &pool, Tensor::ABSENT, Tensor::ABSENT).unwrap();
    }

    // The codes and scales, token by token.
    let code_row = qz.k_codes.width as usize;
    let scale_row = qz.k_scales.width as usize;
    let (kc, vc) = (
        gpu.read_as::<u8>(&qz.k_codes),
        gpu.read_as::<u8>(&qz.v_codes),
    );
    let (ks, vs) = (gpu.read(&qz.k_scales), gpu.read(&qz.v_scales));
    let head_codes = code_row / h;
    let head_scales = scale_row / h;
    let mut shadow_k = vec![0f32; pg.total * ps * row];
    let mut shadow_v = vec![0f32; pg.total * ps * row];
    for r in 0..kv_lens.len() {
        for t in 0..kv_lens[r] as usize {
            let page = pg.indices[pg.indptr[r] as usize + t / ps] as usize;
            let slot = page * ps + t % ps;
            for hh in 0..h {
                let x = &keys[r][(t * h + hh) * d..(t * h + hh + 1) * d];
                let y = &vals[r][(t * h + hh) * d..(t * h + hh + 1) * d];
                let tag = format!("{:?} req {r} token {t} head {hh}", c.scheme);
                let at = slot * code_row + hh * head_codes;
                let sat = slot * scale_row + hh * head_scales;
                for (what, src, codes, scales, shadow) in [
                    ("k", x, &kc, &ks, &mut shadow_k),
                    ("v", y, &vc, &vs, &mut shadow_v),
                ] {
                    let (want_codes, want_scales) = quantize_head(c.scheme, src);
                    let have_codes = &codes[at..at + head_codes];
                    assert_eq!(have_codes, &want_codes[..], "{tag} {what} codes");
                    let have_scales = &scales[sat..sat + want_scales.len()];
                    assert_eq!(have_scales, &want_scales[..], "{tag} {what} scales");
                    let back = dequantize_head(c.scheme, have_codes, have_scales, d);
                    shadow[(slot * h + hh) * d..(slot * h + hh + 1) * d].copy_from_slice(&back);
                }
            }
        }
    }

    // A decode of each request's last token: the shadow it refreshed, and
    // its attention over it.
    let n = kv_lens.len() as u32;
    qz.pool.last_page_lens = gpu.column(&last_page_lens(&kv_lens, c.page_size));
    let q_heads = 2 * c.kv_heads;
    let qv: Vec<f32> = values((n * q_heads * c.head_dim) as usize, 5, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let q = RaggedTensor {
        data: gpu.tensor(&qv, n, q_heads * c.head_dim, Dtype::Bf16),
        indptr: gpu.column(&(0..=n as i32).collect::<Vec<_>>()),
    };
    let shape = plan::Shape {
        num_requests: n,
        lane_offset: 0,
        num_q_heads: q_heads,
        num_kv_heads: c.kv_heads,
        head_dim: c.head_dim,
        page_size: c.page_size,
        hnd_layout: false,
    };
    let live = plan::Live {
        requests: n,
        lane_offset: 0,
        row_offset: 0,
        rows: n,
    };
    let kv_len: Vec<i32> = kv_lens.iter().map(|&x| x as i32).collect();
    let int = gpu.zeros(1, 8 << 20, Dtype::F32);
    let float = gpu.zeros(1, 8 << 20, Dtype::F32);
    let workspace = plan::Workspace {
        int_ptr: int.ptr,
        int_bytes: 32 << 20,
        float_ptr: float.ptr,
        float_bytes: 32 << 20,
    };
    let device = plan::Device::probe(&ctx).unwrap();
    let built = plan::plan_decode(
        &pg.indptr,
        &kv_len,
        shape,
        live,
        None,
        false,
        4096,
        plan::Toggles::default(),
        &device,
        workspace,
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    let mut o = gpu.zeros(n, q_heads * c.head_dim, Dtype::Bf16);
    attn::decode(&ctx, q, &built, &qz.pool, None, c.head_dim, 0.0, &mut o).unwrap();

    let (have_k, have_v) = (gpu.read(&qz.pool.bf16_keys), gpu.read(&qz.pool.bf16_values));
    for &page in &pg.indices {
        let span = page as usize * ps * row..(page as usize + 1) * ps * row;
        for (what, have, want) in [("k", &have_k, &shadow_k), ("v", &have_v, &shadow_v)] {
            for i in span.clone() {
                // Slots past a request's kv hold the seed codes: compare
                // only what the host knows.
                let slot = i / row;
                let filled = (0..kv_lens.len()).any(|r| {
                    (0..kv_lens[r] as usize).any(|t| {
                        pg.indices[pg.indptr[r] as usize + t / ps] as usize * ps + t % ps == slot
                    })
                });
                if filled {
                    assert_eq!(have[i], want[i], "{:?} shadow {what}[{i}]", c.scheme);
                }
            }
        }
    }

    let got = gpu.read(&o);
    let (hq, g) = (q_heads as usize, 2);
    let scale = 1.0 / (d as f64).sqrt();
    for r in 0..n as usize {
        for head in 0..hq {
            let kh = head / g;
            let qrow = &qv[(r * hq + head) * d..(r * hq + head + 1) * d];
            let kv = kv_lens[r] as usize;
            let slot_of =
                |t: usize| pg.indices[pg.indptr[r] as usize + t / ps] as usize * ps + t % ps;
            let s: Vec<f64> = (0..kv)
                .map(|t| {
                    let at = (slot_of(t) * h + kh) * d;
                    (0..d)
                        .map(|j| f64::from(qrow[j]) * f64::from(shadow_k[at + j]))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let m = s.iter().copied().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for j in 0..d {
                let acc: f64 = (0..kv)
                    .map(|t| e[t] * f64::from(shadow_v[(slot_of(t) * h + kh) * d + j]))
                    .sum();
                let want = (acc / z) as f32;
                let have = got[(r * hq + head) * d + j];
                assert!(
                    (have - want).abs() <= 1.5e-2,
                    "{:?} decode req {r} head {head} [{j}] = {have}, want {want}",
                    c.scheme
                );
            }
        }
    }
}

#[test]
fn every_scheme_appends_dequantizes_and_decodes() {
    let Some(gpu) = Gpu::open() else { return };
    for scheme in [
        KvScheme::Fp8PerTensor,
        KvScheme::Fp8PerTokenHead,
        KvScheme::Int8PerTokenHead,
        KvScheme::Fp4Block,
    ] {
        let c = Case {
            scheme,
            kv_heads: 2,
            head_dim: 128,
            page_size: 16,
            // Past 448 the per-tensor codes saturate; below 2^-6 they are
            // subnormal.
            key_scale: if scheme == KvScheme::Fp8PerTensor {
                600.0
            } else {
                1.0
            },
        };
        run(&gpu, &c, &[0, 5, 16, 40], &[3, 1, 20, 9]);
    }
}

/// Under a staged window the append reads the window's plane rows and
/// resolves its slots against the indptr as launched, capped at the
/// window's lanes.
#[test]
fn a_launch_window_appends_the_staged_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let c = Case {
        scheme: KvScheme::Fp8PerTokenHead,
        kv_heads: 1,
        head_dim: 64,
        page_size: 8,
        key_scale: 1.0,
    };
    // Two requests of 3 and 2 fresh tokens; the plane holds 7 rows, the
    // staged rows are plane rows 2..7, and the launch covers all 7.
    let kv_lens = [3u32, 2];
    let pg = pages(&kv_lens, 8);
    let mut qz = quantized(&gpu, &c, &pg);
    qz.pool.last_page_lens = gpu.column(&last_page_lens(&kv_lens, 8));
    let plane: Vec<f32> = values(7 * 64, 4, 1.0).into_iter().map(round_bf16).collect();
    let k = RaggedTensor {
        data: gpu.tensor(&plane, 7, 64, Dtype::Bf16),
        indptr: gpu.column(&[0, 3, 5, 5]),
    };
    let v = gpu.tensor(&plane, 7, 64, Dtype::Bf16);
    let win = gpu.ints(&[5, 2, 2, 0]);
    ctx.arm_stage(win.ptr);
    attn::kv_append(&ctx, k, v, &qz.pool, Tensor::ABSENT, Tensor::ABSENT).unwrap();
    ctx.disarm_stage();
    let codes = gpu.read_as::<u8>(&qz.k_codes);
    let scales = gpu.read(&qz.k_scales);
    for t in 0..5usize {
        let (r, at) = if t < 3 { (0, t) } else { (1, t - 3) };
        let slot = pg.indices[pg.indptr[r] as usize] as usize * 8 + at;
        let src = &plane[(t + 2) * 64..(t + 3) * 64];
        let (want_codes, want_scales) = quantize_head(c.scheme, src);
        assert_eq!(
            &codes[slot * 64..(slot + 1) * 64],
            &want_codes[..],
            "row {t} codes"
        );
        assert_eq!(scales[slot], want_scales[0], "row {t} scale");
    }
}
