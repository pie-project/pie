#![cfg(target_vendor = "apple")]
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{KvPool, PrefillPlan, RaggedTensor, Tensor, attn};
use poem_ir::Dtype;
use crate::common::{bytes, f32s, mask_words, u32s, vals};

/// Head width 128 (Qwen3, Mistral, Phi-3) through the simdgroup-matrix kernel
/// (`sdpa_paged_mma_wide_bfloat16_d_128`: bf16 tiles, f32 scores and output)
/// against the tiled kernel and an fp64 host reference: 35 to
/// 8190 rows, interleaved requests, rows appended to a long cached prefix,
/// plain causal, windowed and masked, and a mask that admits later keys. The
/// reference is computed on every row up to 2048 and on a sample of rows
/// past that (every 61st, and the last 64). Both kernels must sit within the
/// stated tolerance of the reference, the new one no worse than the tiled.
#[test]
fn mma_attention_at_128_matches_the_tiled_kernel_and_fp64() {
    let Ok(dev) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipes = Pipelines::new();
    let mma = engine_metal::chip::tuning("[kernel]\nsdpa.prefill = \"tiled\"", false);
    let tiled = engine_metal::chip::tuning("[kernel]\nsdpa.prefill = \"tiled\"\nsdpa.mma = false", false);
    let d = 128usize;
    // (rows, query heads, kv heads, requests, pages per request, first position, modes)
    let cases: [(usize, usize, usize, usize, usize, usize, &[u8]); 7] = [
        (35, 8, 2, 3, 3, 65, &[0, 1, 2]),
        (300, 4, 1, 2, 12, 0, &[0, 1, 2]),
        (512, 16, 8, 1, 16, 0, &[0, 1, 2]),
        (64, 16, 8, 1, 160, 5000, &[0, 1, 2]),
        (2048, 4, 2, 1, 64, 0, &[0, 1]),
        (4096, 2, 1, 1, 128, 0, &[0, 1]),
        (8190, 2, 1, 1, 256, 0, &[0, 1]),
    ];
    let mut worst = (0f64, 0f64, String::new());
    for (m, qh, kh, requests, np, base, modes) in cases {
        let page = 32usize;
        let slots = np * requests * page;
        let k = vals(slots * kh * d, 7);
        let v = vals(k.len(), 19);
        let q = vals(m * qh * d, 41);
        let owners: Vec<u32> = (0..m).map(|i| (i % requests) as u32).collect();
        let pos: Vec<u32> = (0..m).map(|i| (base + i / requests) as u32).collect();
        let pages: Vec<u32> = (0..(np * requests) as u32).rev().collect();
        let ptr: Vec<u32> = (0..=requests).map(|i| (i * np) as u32).collect();
        let upload = |v: &[u8]| {
            let mut b = Buffer::zeroed(&dev, v.len() as u64).unwrap();
            b.write(0, v).unwrap();
            b
        };
        let kp = upload(&bytes(&k));
        let vp = upload(&bytes(&v));
        let qb = upload(&bytes(&q));
        let pb = upload(&u32s(&pages));
        let ip = upload(&u32s(&ptr));
        let po = upload(&u32s(&pos));
        let ow = upload(&u32s(&owners));
        let qp = upload(&u32s(&[0, m as u32]));
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).unwrap();
        let tensor =
            |b: &Buffer, r: usize, c: usize, t: Dtype| Tensor::new(bind(b), r as u32, c as u32, t);
        let pool = KvPool {
            keys: tensor(&kp, slots, kh * d, Dtype::Bf16),
            values: tensor(&vp, slots, kh * d, Dtype::Bf16),
            page_indices: tensor(&pb, np * requests, 1, Dtype::U32),
            page_indptr: tensor(&ip, requests + 1, 1, Dtype::U32),
            page_size: page as i32,
            seq_stride: (kh * d) as u64,
            head_stride: d as u64,
            q8: false,
            page_stride: [0; 2],
        }
        .paged_densely();
        let sampled: Vec<usize> = (0..m)
            .filter(|&row| m <= 2048 || row % 61 == 0 || row + 64 >= m)
            .collect();
        for &mode in modes {
            let window = if mode == 1 && m < 2048 { Some(19u32) } else { None };
            let mask: Vec<u8> = (0..m * np * page).map(|i| u8::from(i % 7 != 0)).collect();
            let mb = upload(&mask);
            let eb = upload(&mask_words(&vec![u32::from(mode); m]));
            let plan = PrefillPlan {
                positions: tensor(&po, m, 1, Dtype::I32),
                last_page_len: tensor(&po, m, 1, Dtype::I32),
                request_of_token: tensor(&ow, m, 1, Dtype::I32),
                mask: tensor(&mb, mask.len(), 1, Dtype::U8),
                mask_enabled: tensor(&eb, m, 1, Dtype::U32),
                mask_stride: (np * page) as u32,
            };
            let scale = 1. / (d as f32).sqrt();
            let mut reference = vec![0f64; q.len()];
            for &row in &sampled {
                for head in 0..qh {
                    let last = if mode == 2 { np * page - 1 } else { pos[row] as usize };
                    let mut scores = vec![];
                    for t in 0..=last {
                        if window.is_some_and(|w| t + (w as usize) <= pos[row] as usize)
                            || mode != 0 && mask[row * np * page + t] == 0
                        {
                            continue;
                        }
                        let slot =
                            pages[owners[row] as usize * np + t / page] as usize * page + t % page;
                        let offset = (slot * kh + head / (qh / kh)) * d;
                        let dot = (0..d)
                            .map(|j| q[(row * qh + head) * d + j] as f64 * k[offset + j] as f64)
                            .sum::<f64>()
                            * scale as f64;
                        scores.push((dot, offset));
                    }
                    let max = scores.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
                    let sum = scores.iter().map(|x| (x.0 - max).exp()).sum::<f64>();
                    for (s, offset) in scores {
                        let p = (s - max).exp() / sum;
                        for j in 0..d {
                            reference[(row * qh + head) * d + j] += p * v[offset + j] as f64;
                        }
                    }
                }
            }
            let run = |tuning: &kernels_metal::Tuning| -> Vec<f32> {
                let y = upload(&vec![0xa5; q.len() * 2 + 64]);
                let frame = dev.frame().unwrap();
                attn::arbiter::masked(
                    &Sink::new(&dev, &frame, &pipes, &handles),
                    RaggedTensor {
                        data: tensor(&qb, m, qh * d, Dtype::Bf16),
                        indptr: tensor(&qp, 2, 1, Dtype::I32),
                    },
                    &plan,
                    plan.mask,
                    &pool,
                    window,
                    true,
                    d as u32,
                    scale,
                    tensor(&y, m, qh * d, Dtype::Bf16),
                    requests as u32,
                    tuning,
                )
                .unwrap();
                frame.commit().unwrap();
                let raw = handles.read(bind(&y), y.bytes()).unwrap();
                assert!(
                    raw[q.len() * 2..].iter().all(|x| *x == 0xa5),
                    "the kernel wrote past its rows"
                );
                f32s(&raw[..q.len() * 2])
            };
            let (got, old) = (run(&mma), run(&tiled));
            let against = |out: &[f32]| -> (f64, f64) {
                let (mut err, mut norm, mut abs) = (0f64, 0f64, 0f64);
                for &row in &sampled {
                    for at in row * qh * d..(row + 1) * qh * d {
                        let e = out[at] as f64 - reference[at];
                        err += e * e;
                        norm += reference[at] * reference[at];
                        abs = abs.max(e.abs());
                    }
                }
                ((err / norm.max(1e-30)).sqrt(), abs)
            };
            let ((rms, abs), (old_rms, _)) = (against(&got), against(&old));
            let label = format!("rows={m} heads={qh}/{kh} requests={requests} base={base} window={window:?} mode={mode}");
            assert!(rms < 0.008, "d=128 {label}: the mma kernel parts from fp64 by relative rms {rms}");
            assert!(abs < 0.03, "d=128 {label}: the mma kernel's worst element parts from fp64 by {abs}");
            assert!(
                rms <= old_rms * 1.25 + 1e-4,
                "d=128 {label}: the mma kernel ({rms}) is worse than the tiled ({old_rms})"
            );
            if abs > worst.0 {
                worst = (abs, rms, label);
            }
        }
    }
}
