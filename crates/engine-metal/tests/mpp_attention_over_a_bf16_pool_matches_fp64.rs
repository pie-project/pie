#![cfg(target_vendor = "apple")]
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{KvPool, PrefillPlan, RaggedTensor, Tensor, attn};
use poem_ir::Dtype;

fn bf(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
fn vals(n: usize, s: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ s;
            x ^= x >> 33;
            x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
            bf(((x >> 40) as f32 / 16777216. - 0.5) * 4.)
        })
        .collect()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect()
}

#[test]
fn mpp_attention_over_a_bf16_pool_matches_fp64() {
    let Ok(dev) = Context::bind() else {
        return;
    };
    // MPP ships with Metal 4, which is what the elastic pool asks of the device too.
    if !dev.supports_elastic() {
        return;
    }
    let handles = Handles::new();
    let pipes = Pipelines::new();
    let tuning = kernels_metal::DeviceTuning {
        sdpa_mpp: true,
        ..Default::default()
    };
    // (head dim, rows, query heads, kv heads, requests, pages per request, first position)
    for (d, m, qh, kh, requests, np, base) in [
        (128usize, 35usize, 6usize, 2usize, 3usize, 3usize, 65usize),
        (256, 35, 6, 2, 3, 3, 65),
        (256, 64, 8, 4, 2, 4, 65),
        (256, 600, 2, 1, 3, 10, 65),
    ] {
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
        let y = upload(&vec![0xa5; q.len() * 2 + 64]);
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
        };
        // mode 0 is plain causal, 1 a sparse mask, 2 a mask that also admits later keys.
        for (window, mode) in [(None, 0u8), (Some(19u32), 1), (None, 1), (None, 2)] {
            let mask: Vec<u8> = (0..m * np * page).map(|i| u8::from(i % 7 != 0)).collect();
            let mb = upload(&mask);
            let eb = upload(&vec![mode; m]);
            let plan = PrefillPlan {
                positions: tensor(&po, m, 1, Dtype::I32),
                request_of_token: tensor(&ow, m, 1, Dtype::I32),
                mask: tensor(&mb, mask.len(), 1, Dtype::U8),
                mask_enabled: tensor(&eb, m, 1, Dtype::U8),
                mask_stride: (np * page) as u32,
            };
            let scale = 1. / (d as f32).sqrt();
            let mut reference = vec![0f64; q.len()];
            for row in 0..m {
                for head in 0..qh {
                    let last = if mode == 2 {
                        np * page - 1
                    } else {
                        pos[row] as usize
                    };
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
                &tuning,
            )
            .unwrap();
            frame.commit().unwrap();
            let raw = handles.read(bind(&y), y.bytes()).unwrap();
            assert!(
                raw[q.len() * 2..].iter().all(|x| *x == 0xa5),
                "the kernel wrote past its rows"
            );
            let got = f32s(&raw[..q.len() * 2]);
            let rms = (got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (*a as f64 - b).powi(2))
                .sum::<f64>()
                / reference.iter().map(|b| b * b).sum::<f64>().max(1e-30))
            .sqrt();
            eprintln!("d={d} rows={m} window={window:?} mode={mode}: relative rms {rms:.6}");
            assert!(
                rms < 0.008,
                "d={d} rows={m} window={window:?} mode={mode}: relative rms {rms}"
            );
        }
    }
}
