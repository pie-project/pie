#![cfg(target_vendor = "apple")]
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{DecodePlan, KvPool, PrefillPlan, RaggedTensor, Tensor, attn};
use model_ir::Dtype;
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
fn stored_q8_and_all_attention_routes_match_fp64() {
    if std::env::var("PIE_METAL_KV_Q8").as_deref() != Ok("1") {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "stored_q8_and_all_attention_routes_match_fp64",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("PIE_METAL_KV_Q8", "1")
            .status()
            .unwrap();
        assert!(status.success(), "isolated Q8 integration test failed");
        return;
    }
    let dev = Context::bind().unwrap();
    let handles = Handles::new();
    let pipes = Pipelines::new();
    for (d, m, qh, kh, requests, np, base, page) in [
        (
            64usize, 35usize, 6usize, 2usize, 3usize, 3usize, 65usize, 32usize,
        ),
        (128, 35, 6, 2, 3, 3, 65, 32),
        (256, 35, 6, 2, 3, 3, 65, 32),
        (512, 35, 6, 2, 3, 3, 65, 32),
        (256, 8, 24, 4, 1, 130, 4096, 32),
        (256, 35, 6, 2, 3, 6, 65, 16),
        (256, 512, 2, 1, 1, 19, 65, 32),
        (256, 32, 24, 4, 4, 5, 65, 32),
    ] {
        let slots = np * requests * page;
        let mut k = vals(slots * kh * d, 7);
        let mut v = vals(k.len(), 19);
        k[..d].fill(0.);
        v[..d].fill(0.);
        for a in [&mut k, &mut v] {
            a[d..2 * d].fill(0.);
            a[d] = 127.;
            for (i, z) in [0.5, -0.5, 1.5, -1.5, 2.5, -2.5].into_iter().enumerate() {
                a[d + 1 + i] = z;
            }
        }
        let q = vals(m * qh * d, 41);
        let owners: Vec<u32> = (0..m).map(|i| (i % requests) as u32).collect();
        let pos: Vec<u32> = (0..m).map(|i| (base + i / requests) as u32).collect();
        let pages: Vec<u32> = (0..(np * requests) as u32).rev().collect();
        let ptr: Vec<u32> = (0..=requests).map(|i| (i * np) as u32).collect();
        let wp: Vec<u32> = (0..slots).map(|i| (i / page) as u32).collect();
        let wo: Vec<u32> = (0..slots).map(|i| (i % page) as u32).collect();
        let upload = |v: &[u8]| {
            let mut b = Buffer::zeroed(&dev, v.len() as u64).unwrap();
            b.write(0, v).unwrap();
            b
        };
        let mut kp = upload(&vec![0xa5; k.len() * 2 + 64]);
        let mut vp = upload(&vec![0xa5; k.len() * 2 + 64]);
        let kb = upload(&bytes(&k));
        let vb = upload(&bytes(&v));
        let qb = upload(&bytes(&q));
        let y = upload(&vec![0xa5; q.len() * 2 + 64]);
        let pb = upload(&u32s(&pages));
        let ip = upload(&u32s(&ptr));
        let wb = upload(&u32s(&wp));
        let ob = upload(&u32s(&wo));
        let po = upload(&u32s(&pos));
        let ow = upload(&u32s(&owners));
        let lse = upload(&vec![0xa5; m * qh * 4 + 64]);
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
        let frame = dev.frame().unwrap();
        attn::kv_append(
            &Sink::new(&dev, &frame, &pipes, &handles),
            tensor(&kb, slots, kh * d, Dtype::Bf16),
            tensor(&vb, slots, kh * d, Dtype::Bf16),
            &pool,
            tensor(&wb, slots, 1, Dtype::U32),
            tensor(&ob, slots, 1, Dtype::U32),
        )
        .unwrap();
        frame.commit().unwrap();
        let mut kd = vec![0f32; k.len()];
        let mut vd = kd.clone();
        for (orig, buf, qd) in [(&k, &kp, &mut kd), (&v, &vp, &mut vd)] {
            let raw = handles.read(bind(buf), buf.bytes()).unwrap();
            assert!(raw[orig.len() * 2..].iter().all(|x| *x == 0xa5));
            for (row, input) in orig.chunks_exact(d).enumerate() {
                let maximum = input.iter().map(|x| x.abs()).fold(0f32, f32::max);
                let scale = maximum / 127.;
                let at = row * 2 * d;
                let actual = f32::from_le_bytes(raw[at + d..at + d + 4].try_into().unwrap());
                assert!((actual - scale).abs() < 1e-8);
                for j in 0..d {
                    let code = if maximum == 0. {
                        0
                    } else {
                        (input[j] * (127. / maximum))
                            .round_ties_even()
                            .clamp(-127., 127.) as i8
                    };
                    let gpu = raw[at + j] as i8;
                    if row == 1 {
                        assert_eq!(gpu, code, "exact nearest-even tie fixture");
                    }
                    if gpu != code {
                        let ideal = if maximum == 0. {
                            0.
                        } else {
                            input[j] as f64 * 127. / maximum as f64
                        };
                        assert!(
                            (gpu as f64 - ideal).abs() <= 0.50002
                                && (gpu as i32 - code as i32).abs() <= 1,
                            "store d={d} row={row} j={j} ideal={ideal} cpu={code} gpu={gpu}"
                        );
                    }
                    qd[row * d + j] = (gpu as f32) * actual;
                }
                assert!(raw[at + d + 4..at + 2 * d].iter().all(|x| *x == 0xa5));
            }
        }
        let from = 32usize;
        let to = 95usize;
        let cell = kh * d * 2;
        for b in [&mut kp, &mut vp] {
            let raw = handles.read(bind(b), b.bytes()).unwrap();
            b.write((to * cell) as u64, &raw[from * cell..(from + 1) * cell])
                .unwrap();
        }
        for v in [&mut kd, &mut vd] {
            v.copy_within(from * kh * d..(from + 1) * kh * d, to * kh * d);
        }
        for (causal, window, mode) in [
            (true, None, 0u8),
            (true, Some(19), 1),
            (true, None, 2),
            (true, None, 3),
            (false, Some(19), 1),
        ] {
            let mask: Vec<u8> = (0..m * np * page)
                .map(|i| if mode == 3 { 0 } else { u8::from(i % 7 != 0) })
                .collect();
            let mb = upload(&mask);
            let eb = upload(&vec![if mode == 3 { 1 } else { mode }; m]);
            let plan = DecodePlan {
                positions: tensor(&po, m, 1, Dtype::I32),
                request_of_token: tensor(&ow, m, 1, Dtype::I32),
                mask: tensor(&mb, mask.len(), 1, Dtype::U8),
                mask_enabled: tensor(&eb, m, 1, Dtype::U8),
                mask_stride: (np * page) as u32,
            };
            let pre = PrefillPlan {
                positions: plan.positions,
                request_of_token: plan.request_of_token,
                mask: plan.mask,
                mask_enabled: plan.mask_enabled,
                mask_stride: plan.mask_stride,
            };
            let scale = 1. / (d as f32).sqrt();
            let mut reference = vec![0f64; q.len()];
            let mut refs = vec![f64::NEG_INFINITY; m * qh];
            for row in 0..m {
                for head in 0..qh {
                    let mut scores = vec![];
                    let last = if !causal || mode == 2 {
                        np * page - 1
                    } else {
                        pos[row] as usize
                    };
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
                            .map(|j| q[(row * qh + head) * d + j] as f64 * kd[offset + j] as f64)
                            .sum::<f64>()
                            * scale as f64;
                        scores.push((dot, offset));
                    }
                    if !scores.is_empty() {
                        let max = scores.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
                        let sum = scores.iter().map(|x| (x.0 - max).exp()).sum::<f64>();
                        refs[row * qh + head] = (max + sum.ln()) / std::f64::consts::LN_2;
                        for (s, offset) in scores {
                            let p = (s - max).exp() / sum;
                            for j in 0..d {
                                reference[(row * qh + head) * d + j] += p * vd[offset + j] as f64;
                            }
                        }
                    }
                }
            }
            let words = m.div_ceil(8) as u32
                * attn::split::workspace_words(attn::split::split_count(m as u32));
            let workspace = upload(&vec![0xa5; words as usize * 4 + 64]);
            for route in 0..5 {
                if !causal && route != 1 {
                    continue;
                }
                let frame = dev.frame().unwrap();
                let sink = Sink::new(&dev, &frame, &pipes, &handles);
                let qt = tensor(&qb, m, qh * d, Dtype::Bf16);
                let yt = tensor(&y, m, qh * d, Dtype::Bf16);
                if route == 4 {
                    if d != 256 || qh != 24 || window.is_some() {
                        continue;
                    }
                    assert!(
                        attn::split::try_paged(
                            &sink,
                            qt,
                            &pool,
                            &pre,
                            plan.mask,
                            window,
                            causal,
                            d as u32,
                            scale,
                            yt,
                            requests as u32,
                            &|rows, width| (rows == 1 && width <= words).then(|| tensor(
                                &workspace,
                                1,
                                width as usize,
                                Dtype::F32
                            ))
                        )
                        .unwrap()
                    );
                } else if route == 0 {
                    attn::decode(&sink, qt, &plan, &pool, window, d as u32, scale, yt).unwrap();
                } else if route == 1 {
                    let tuning = kernels_metal::DeviceTuning {
                        sdpa_mpp: true,
                        ..Default::default()
                    };
                    attn::arbiter::masked(
                        &sink,
                        RaggedTensor {
                            data: qt,
                            indptr: tensor(&qp, 2, 1, Dtype::I32),
                        },
                        &pre,
                        plan.mask,
                        &pool,
                        window,
                        causal,
                        d as u32,
                        scale,
                        yt,
                        requests as u32,
                        &tuning,
                    )
                    .unwrap();
                } else if route == 3 {
                    attn::prefill(
                        &sink,
                        RaggedTensor {
                            data: qt,
                            indptr: tensor(&qp, 2, 1, Dtype::I32),
                        },
                        &pre,
                        &pool,
                        window,
                        d as u32,
                        kh as u32,
                        scale,
                        yt,
                    )
                    .unwrap();
                } else {
                    attn::decode_lse(
                        &sink,
                        qt,
                        &plan,
                        &pool,
                        window,
                        d as u32,
                        scale,
                        yt,
                        tensor(&lse, m, qh, Dtype::F32),
                    )
                    .unwrap();
                }
                frame.commit().unwrap();
                let scratch = handles.read(bind(&workspace), workspace.bytes()).unwrap();
                assert!(scratch[words as usize * 4..].iter().all(|x| *x == 0xa5));
                let raw = handles.read(bind(&y), y.bytes()).unwrap();
                assert!(raw[q.len() * 2..].iter().all(|x| *x == 0xa5));
                let got = f32s(&raw[..q.len() * 2]);
                assert!(got.iter().all(|v| v.is_finite()));
                let rms = (got
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (*a as f64 - b).powi(2))
                    .sum::<f64>()
                    / reference.iter().map(|b| b * b).sum::<f64>().max(1e-30))
                .sqrt();
                assert!(rms < 0.008, "d={d} route={route} mode={mode} rms={rms}");
                if route == 2 {
                    let raw = handles.read(bind(&lse), lse.bytes()).unwrap();
                    assert!(raw[m * qh * 4..].iter().all(|x| *x == 0xa5));
                    for (b, r) in raw[..m * qh * 4].as_chunks::<4>().0.iter().zip(&refs) {
                        let x = f32::from_le_bytes(*b) as f64;
                        assert!(x == *r || (x - r).abs() < 2e-4, "lse{x} vs{r}");
                    }
                }
                eprintln!(
                    "Q8CHECK d={d} m={m} page={page} route={route} mode={mode} causal={causal} rms={rms:.8}"
                );
            }
        }
    }
}
