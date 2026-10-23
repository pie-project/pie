//! Paged attention on the first Vulkan device against an f64 host reference, beside the hand-written kernels:
//! decode a row a workgroup on the subgroup tier and the baseline, 256 and 128 invocations, plain, windowed,
//! masked and with its LSE; prefill tiled at 32, 16 and 8 rows, a request
//! straddling a tile, with and without its LSE.

use kernels_vulkan::{KvPool, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

fn tuning(gpu: &Gpu) -> kernels_vulkan::Tuning {
    let (profile, _) = engine_vulkan::chip::resolve(&gpu.device.chip(), None);
    kernels_vulkan::Tuning::new(profile, kernels_vulkan::tuning::Limits::ASSUMED)
}

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

const PAGE: usize = 16;
const QH: usize = 8;
const KVH: usize = 2;

struct Batch {
    positions: Vec<i32>,
    owner: Vec<i32>,
    /// Row `r` reads its mask when `masks[r]` is some, `mask_stride` bytes.
    masks: Vec<Option<Vec<u8>>>,
}

/// Request 0 reads pages [3, 0, 5], request 1 pages [1, 2].
const PAGES: [u32; 5] = [3, 0, 5, 1, 2];
const INDPTR: [u32; 3] = [0, 3, 5];
const MASK_STRIDE: usize = 48;

fn slot(req: usize, pos: usize) -> usize {
    PAGES[INDPTR[req] as usize + pos / PAGE] as usize * PAGE + pos % PAGE
}

/// Each row's attention, causal over its request's keys, inside the window
/// and the row's mask, and its base-2 LSE.
fn reference(
    b: &Batch,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    hd: usize,
    scale: f64,
    window: Option<usize>,
) -> (Vec<f64>, Vec<f64>) {
    let rows = b.positions.len();
    let mut o = vec![0.0; rows * QH * hd];
    let mut l = vec![0.0; rows * QH];
    for r in 0..rows {
        let pos = b.positions[r] as usize;
        let start = window.map_or(0, |w| (pos + 1).saturating_sub(w));
        let keys: Vec<usize> = (start..=pos)
            .filter(|&kp| {
                b.masks[r]
                    .as_ref()
                    .is_none_or(|m| kp < MASK_STRIDE && m[kp] != 0)
            })
            .collect();
        for h in 0..QH {
            let kh = h / (QH / KVH);
            let qv = &q[(r * QH + h) * hd..][..hd];
            let at = |kp: usize| (slot(b.owner[r] as usize, kp) * KVH + kh) * hd;
            let s: Vec<f64> = keys
                .iter()
                .map(|&kp| {
                    qv.iter()
                        .zip(&k[at(kp)..][..hd])
                        .map(|(a, c)| a * c)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - top).exp()).sum();
            l[r * QH + h] = (z.ln() + top) * std::f64::consts::LOG2_E;
            for d in 0..hd {
                o[(r * QH + h) * hd + d] = keys
                    .iter()
                    .zip(&s)
                    .map(|(&kp, x)| (x - top).exp() * v[at(kp) + d])
                    .sum::<f64>()
                    / z;
            }
        }
    }
    (o, l)
}

#[test]
fn paged_attention_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let slots = 6 * PAGE;
    let (pi, pp) = (gpu.up_u32(&PAGES), gpu.up_u32(&INDPTR));
    let decode = Batch {
        positions: vec![39, 20, 5, 30],
        owner: vec![0, 1, 0, 0],
        masks: vec![
            None,
            None,
            None,
            Some((0..MASK_STRIDE).map(|i| u8::from(i % 3 != 1)).collect()),
        ],
    };
    let prefill = Batch {
        positions: (0..26).chain(0..11).collect(),
        owner: std::iter::repeat_n(0, 26)
            .chain(std::iter::repeat_n(1, 11))
            .collect(),
        masks: vec![None; 37],
    };
    for hd in [64usize, 128] {
        let scale = 1.0 / (hd as f64).sqrt();
        let (keys, kr) = gpu.up(&noise(hd as u64, slots * KVH * hd));
        let (values, vr) = gpu.up(&noise(hd as u64 + 1, slots * KVH * hd));
        let (kr, vr) = (f64s(&kr), f64s(&vr));
        let pool = KvPool {
            keys: t(keys, slots, KVH * hd, Dtype::Bf16),
            values: t(values, slots, KVH * hd, Dtype::Bf16),
            page_indices: t(pi, PAGES.len(), 1, Dtype::U32),
            page_indptr: t(pp, 3, 1, Dtype::U32),
            page_size: PAGE as i32,
            seq_stride: (KVH * hd) as u64,
            head_stride: hd as u64,
        };
        let spool = kernels_slang::KvPool {
            keys: st(pool.keys),
            values: st(pool.values),
            page_indices: st(pool.page_indices),
            page_indptr: st(pool.page_indptr),
            page_size: pool.page_size,
            seq_stride: pool.seq_stride,
            head_stride: pool.head_stride,
        };
        for (what, batch, prefilling) in [("decode", &decode, false), ("prefill", &prefill, true)] {
            let rows = batch.positions.len();
            let (q, qr) = gpu.up(&noise(hd as u64 + rows as u64, rows * QH * hd));
            let qr = f64s(&qr);
            let pos = gpu.up_i32(&batch.positions);
            let own = gpu.up_i32(&batch.owner);
            let mut mask = vec![0u8; rows * MASK_STRIDE];
            let mut enabled = vec![0u8; rows];
            for (r, m) in batch.masks.iter().enumerate() {
                if let Some(m) = m {
                    mask[r * MASK_STRIDE..][..MASK_STRIDE].copy_from_slice(m);
                    enabled[r] = 1;
                }
            }
            let pack = |b: &[u8]| -> Vec<u32> {
                b.chunks(4)
                    .map(|c| {
                        u32::from_le_bytes([
                            c[0],
                            *c.get(1).unwrap_or(&0),
                            *c.get(2).unwrap_or(&0),
                            *c.get(3).unwrap_or(&0),
                        ])
                    })
                    .collect()
            };
            let (mb, eb) = (gpu.up_u32(&pack(&mask)), gpu.up_u32(&pack(&enabled)));
            let tables = kernels_slang::attn::sdpa::Tables {
                positions: st(t(pos, rows, 1, Dtype::I32)),
                request_of_token: st(t(own, rows, 1, Dtype::I32)),
                mask: st(t(mb, rows, MASK_STRIDE, Dtype::U8)),
                mask_enabled: st(t(eb, rows, 1, Dtype::U8)),
                mask_stride: MASK_STRIDE as u32,
            };
            let n = rows * QH * hd;
            let qt = t(q, rows, QH * hd, Dtype::Bf16);
            for window in [None, Some(16usize)] {
                let (want_o, want_l) = reference(batch, &qr, &kr, &vr, hd, scale, window);
                let win = window.map(|w| w as u32);
                let shape = format!("{what} d {hd} window {window:?}");
                {
                    let tn = tuning(&gpu);
                    let o = gpu.zeros(n);
                    let pt = t(pos, rows, 1, Dtype::I32);
                    let ow = t(own, rows, 1, Dtype::I32);
                    let mt = t(mb, rows, MASK_STRIDE, Dtype::U8);
                    let et = t(eb, rows, 1, Dtype::U8);
                    let ot = t(o, rows, QH * hd, Dtype::Bf16);
                    if prefilling {
                        let plan = kernels_vulkan::PrefillPlan {
                            positions: pt,
                            request_of_token: ow,
                            mask: mt,
                            mask_enabled: et,
                            mask_stride: MASK_STRIDE as u32,
                        };
                        let qi = gpu.up_i32(&[0, 26, 37]);
                        let rag = kernels_vulkan::RaggedTensor {
                            data: qt,
                            indptr: t(qi, 3, 1, Dtype::I32),
                        };
                        gpu.run(|c| {
                            kernels_vulkan::attn::arbiter::prefill(
                                c,
                                rag,
                                &plan,
                                &pool,
                                win,
                                hd as u32,
                                KVH as u32,
                                scale as f32,
                                ot,
                                2,
                                &tn,
                            )
                        });
                    } else {
                        let plan = kernels_vulkan::DecodePlan {
                            positions: pt,
                            request_of_token: ow,
                            mask: mt,
                            mask_enabled: et,
                            mask_stride: MASK_STRIDE as u32,
                        };
                        gpu.run(|c| {
                            kernels_vulkan::attn::decode(
                                c,
                                &tn,
                                qt,
                                &plan,
                                &pool,
                                win,
                                hd as u32,
                                scale as f32,
                                ot,
                                None,
                            )
                        });
                    }
                    close(&gpu.down(o, n), &want_o, &shape);
                }

                let tiers: &[(bool, u32, u32)] = if prefilling {
                    &[(false, 256, 32), (false, 256, 16), (false, 256, 8)]
                } else {
                    &[
                        (true, 256, 32),
                        (true, 128, 32),
                        (false, 256, 32),
                        (false, 128, 32),
                    ]
                };
                for &(subgroup, threads, tile) in tiers {
                    let tier = kernels_slang::attn::sdpa::Tier {
                        subgroup,
                        narrow: gpu.device.chip().subgroup < 32,
                        decode_threads: threads,
                        tile_rows: tile,
                    };
                    for with_lse in [false, true] {
                        let (o, l) = (gpu.zeros(n), gpu.up_f32(&vec![0.0; rows * QH]));
                        let lse = with_lse.then(|| st(t(l, rows, QH, Dtype::F32)));
                        let ot = st(t(o, rows, QH * hd, Dtype::Bf16));
                        gpu.run_slang(|c| {
                            if prefilling {
                                kernels_slang::attn::sdpa::prefill(
                                    c,
                                    tier,
                                    st(qt),
                                    &tables,
                                    &spool,
                                    win,
                                    hd as u32,
                                    scale as f32,
                                    ot,
                                    lse,
                                )
                            } else {
                                kernels_slang::attn::sdpa::decode(
                                    c,
                                    tier,
                                    st(qt),
                                    &tables,
                                    &spool,
                                    win,
                                    hd as u32,
                                    scale as f32,
                                    ot,
                                    lse,
                                )
                            }
                        });
                        let label = format!(
                            "slang {shape} subgroup {subgroup} threads {threads} rows {tile} lse {with_lse}"
                        );
                        close(&gpu.down(o, n), &want_o, &label);
                        if with_lse {
                            close(
                                &gpu.down_f32(l, rows * QH),
                                &want_l,
                                &format!("{label}: lse"),
                            );
                        }
                    }
                }
            }
        }
    }
}
