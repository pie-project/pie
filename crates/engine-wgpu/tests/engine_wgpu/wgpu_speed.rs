//! Speed on the first wgpu adapter: kernels-wgpu's hand-written kernels
//! beside kernels-slang's, the same operands, the Slang results checked
//! against the hand-written ones: the affine vector matmul (one activation
//! row a workgroup before, one to eight after), the K-quant matmuls at the
//! baseline tier, and paged decode attention at the baseline tier. Ignored:
//! it asks for a real GPU (`cargo test wgpu_speed -- --ignored
//! --nocapture`). Each kernel fires `REPS` times a frame; the best of
//! `TRIALS` frames is reported.

use std::time::Instant;

use kernels_slang::Tier;
use kernels_slang::linear::kquant::{Scheme, Tactic};
use kernels_slang::linear::qmm::{self, Qmm};
use kernels_wgpu::{Arg, Bank, DecodePlan, Encode, Fire, KvPool, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

const REPS: usize = 50;
const TRIALS: usize = 7;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn t(b: u32, r: usize, w: usize, d: Dtype) -> Tensor {
    Tensor::new(b, r as u32, w as u32, d)
}

/// The best trial's microseconds a dispatch.
fn time(gpu: &Gpu, run: &dyn Fn(&Gpu)) -> f64 {
    run(gpu);
    (0..TRIALS)
        .map(|_| {
            let t0 = Instant::now();
            run(gpu);
            t0.elapsed().as_secs_f64() * 1e6 / REPS as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn pct(slang: f64, hand: f64) -> String {
    format!("{slang:.1} us ({:+.1}%)", (slang / hand - 1.0) * 100.0)
}

fn words(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u32
        })
        .collect()
}

fn qmv(gpu: &mut Gpu) {
    let (n, k, group) = (4096usize, 4096usize, 64usize);
    let w = gpu.up_u32(&words(1, n * k / 8));
    let groups = n * k / group;
    let sc = gpu
        .up(&noise(2, groups)
            .iter()
            .map(|v| v * 0.01)
            .collect::<Vec<_>>())
        .0;
    let bi = gpu
        .up(&noise(3, groups)
            .iter()
            .map(|v| v * 0.01)
            .collect::<Vec<_>>())
        .0;
    let bank = Bank {
        codes: t(w, n, k / 8, Dtype::U32),
        scales: t(sc, n, k / group, Dtype::Bf16),
        biases: Some(t(bi, n, k / group, Dtype::Bf16)),
        group: group as u32,
        bits: 4,
    };
    let sbank = kernels_slang::Bank {
        codes: st(bank.codes),
        scales: st(bank.scales),
        biases: bank.biases.map(st),
        group: bank.group,
        bits: 4,
    };
    for m in [1usize, 2, 4, 8, 11] {
        let xt = t(gpu.up(&noise(4, m * k)).0, m, k, Dtype::Bf16);
        let ht = t(gpu.zeros(m * n), m, n, Dtype::Bf16);
        let hand = time(gpu, &|g| {
            g.run(|c| {
                for _ in 0..REPS {
                    c.fire(
                        Fire::at("quant/qmv.wgsl", "affine_qmv_bf16_gs_64_b_4")
                            .groups([m as u32, (n as u32).div_ceil(8), 1])
                            .group([32, 2, 1]),
                        &[
                            bank.codes.arg(),
                            bank.scales.arg(),
                            bank.biases.expect("affine").arg(),
                            xt.arg(),
                            ht.arg_mut(),
                            (k as i32).arg(),
                            (n as i32).arg(),
                        ],
                    )?;
                }
                Ok(())
            })
        });
        let want: Vec<f64> = gpu
            .down(ht.buf, m * n)
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let mut line = format!("qmv 4-bit g64 {m:>2} x {n} x {k}: hand {hand:.1} us");
        for vecs in [1u32, 2, 4, 8] {
            let yt = t(gpu.zeros(m * n), m, n, Dtype::Bf16);
            let slang = time(gpu, &|g| {
                g.run_slang(|c| {
                    for _ in 0..REPS {
                        qmm::matmul(c, &[Qmm::Qmv { vecs }], st(xt), sbank, st(yt), m as u32)?;
                    }
                    Ok(())
                })
            });
            close(
                &gpu.down(yt.buf, m * n),
                &want,
                &format!("qmv {m} rows, {vecs} a workgroup"),
            );
            line += &format!(", v{vecs} {}", pct(slang, hand));
        }
        eprintln!("{line}");
    }
}

/// Random words with every super-block's scales a small f16.
fn kquant_weights(scheme: Scheme, blocks: usize) -> Vec<u32> {
    let width = scheme.block_bytes() as usize;
    let mut bytes: Vec<u8> = words(9, (blocks * width).div_ceil(4))
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .take(blocks * width)
        .collect();
    let scales: &[usize] = match scheme {
        Scheme::Q2K => &[80, 82],
        Scheme::Q3K => &[108],
        Scheme::Q4K | Scheme::Q5K => &[0, 2],
        Scheme::Q6K => &[208],
    };
    for block in bytes.chunks_mut(width) {
        for at in scales {
            block[*at..*at + 2].copy_from_slice(&0x1400u16.to_le_bytes());
        }
    }
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn kquant(gpu: &mut Gpu) {
    let (n, k) = (4096usize, 4096usize);
    for scheme in [Scheme::Q4K, Scheme::Q6K, Scheme::Q3K] {
        let row_bytes = k / 256 * scheme.block_bytes() as usize;
        let wt = t(
            gpu.up_u32(&kquant_weights(scheme, n * k / 256)),
            n,
            row_bytes,
            Dtype::U8,
        );
        for m in [1usize, 16] {
            let xt = t(gpu.up(&noise(3, m * k)).0, m, k, Dtype::Bf16);
            let ht = t(gpu.zeros(m * n), m, n, Dtype::Bf16);
            let hand = time(gpu, &|g| {
                g.run(|c| {
                    for _ in 0..REPS {
                        kernels_wgpu::linear::kquant::matmul(c, xt, wt, ht)?;
                    }
                    Ok(())
                })
            });
            let want: Vec<f64> = gpu
                .down(ht.buf, m * n)
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let mut line = format!("{scheme:?} m {m:>2}: hand {hand:.1} us");
            for slices in [2u32, 4] {
                let tactic = Tactic {
                    tier: Tier::Baseline,
                    slices,
                };
                let yt = t(gpu.zeros(m * n), m, n, Dtype::Bf16);
                let slang = time(gpu, &|g| {
                    g.run_slang(|c| {
                        for _ in 0..REPS {
                            kernels_slang::linear::kquant::matmul(
                                c,
                                tactic,
                                st(xt),
                                st(wt),
                                st(yt),
                            )?;
                        }
                        Ok(())
                    })
                });
                close(
                    &gpu.down(yt.buf, m * n),
                    &want,
                    &format!("{scheme:?} m {m} s{slices}"),
                );
                line += &format!(", s{slices} {}", pct(slang, hand));
            }
            eprintln!("{line}");
        }
    }
}

fn decode(gpu: &mut Gpu, tuning: &kernels_wgpu::Tuning) {
    const PAGE: usize = 16;
    const QH: usize = 32;
    const KVH: usize = 8;
    const HD: usize = 128;
    let scale = 1.0 / (HD as f32).sqrt();
    for kv in [1024usize, 8192] {
        let pages = kv.div_ceil(PAGE);
        let slots = pages * PAGE;
        let page_ids: Vec<u32> = (0..pages as u32).rev().collect();
        let (pi, pp) = (gpu.up_u32(&page_ids), gpu.up_u32(&[0, pages as u32]));
        let keys = gpu.up(&noise(1, slots * KVH * HD)).0;
        let values = gpu.up(&noise(2, slots * KVH * HD)).0;
        let pool = KvPool {
            keys: t(keys, slots, KVH * HD, Dtype::Bf16),
            values: t(values, slots, KVH * HD, Dtype::Bf16),
            page_indices: t(pi, pages, 1, Dtype::U32),
            page_indptr: t(pp, 2, 1, Dtype::U32),
            page_size: PAGE as i32,
            seq_stride: (KVH * HD) as u64,
            head_stride: HD as u64,
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
        for rows in [1usize, 16] {
            let q = gpu.up(&noise(3, rows * QH * HD)).0;
            let pos = gpu.up_i32(&vec![kv as i32 - 1; rows]);
            let own = gpu.up_i32(&vec![0; rows]);
            let none = gpu.up_u32(&vec![0; rows.div_ceil(4)]);
            let (qt, ot) = (
                t(q, rows, QH * HD, Dtype::Bf16),
                t(gpu.zeros(rows * QH * HD), rows, QH * HD, Dtype::Bf16),
            );
            let (pt, ow, mt) = (
                t(pos, rows, 1, Dtype::I32),
                t(own, rows, 1, Dtype::I32),
                t(none, rows, 1, Dtype::U8),
            );
            let plan = DecodePlan {
                positions: pt,
                request_of_token: ow,
                mask: mt,
                mask_enabled: mt,
                mask_stride: 0,
            };
            let tables = kernels_slang::attn::sdpa::Tables {
                positions: st(pt),
                request_of_token: st(ow),
                mask: st(mt),
                mask_enabled: st(mt),
                mask_stride: 0,
            };
            let hand = time(gpu, &|g| {
                g.run(|c| {
                    for _ in 0..REPS {
                        kernels_wgpu::attn::decode(
                            c, tuning, qt, &plan, &pool, None, HD as u32, scale, ot, None,
                        )?;
                    }
                    Ok(())
                })
            });
            let n = rows * QH * HD;
            let want: Vec<f64> = gpu.down(ot.buf, n).iter().map(|v| f64::from(*v)).collect();
            let mut line = format!("SDPA decode kv {kv:>5} rows {rows:>2}: hand {hand:.1} us");
            for threads in [256u32, 128] {
                let tactic = kernels_slang::attn::sdpa::Tactic {
                    tier: Tier::Baseline,
                    narrow: false,
                    decode_threads: threads,
                    tile_rows: 32,
                };
                let yt = t(gpu.zeros(n), rows, QH * HD, Dtype::Bf16);
                let slang = time(gpu, &|g| {
                    g.run_slang(|c| {
                        for _ in 0..REPS {
                            kernels_slang::attn::sdpa::decode(
                                c,
                                tactic,
                                st(qt),
                                &tables,
                                &spool,
                                None,
                                HD as u32,
                                scale,
                                st(yt),
                                None,
                            )?;
                        }
                        Ok(())
                    })
                });
                close(
                    &gpu.down(yt.buf, n),
                    &want,
                    &format!("decode kv {kv} rows {rows} t{threads}"),
                );
                line += &format!(", baseline t{threads} {}", pct(slang, hand));
            }
            eprintln!("{line}");
        }
    }
}

#[test]
#[ignore = "a benchmark: a real GPU"]
fn wgpu_speed() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let chip = gpu.device.chip();
    eprintln!(
        "adapter: {} (subgroups of 32: {})",
        chip.name,
        chip.subgroups_32()
    );
    let (profile, _) = engine_wgpu::chip::resolve(&chip, None);
    let tuning =
        kernels_wgpu::Tuning::new(profile, gpu.device.cores(), gpu.device.storage_buffers());
    qmv(&mut gpu);
    kquant(&mut gpu);
    decode(&mut gpu, &tuning);
}
