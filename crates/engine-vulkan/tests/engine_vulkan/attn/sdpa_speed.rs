//! Paged attention's speed on the first Vulkan device: the hand-written
//! decode and prefill beside kernels-slang's tiers, the same shapes and the
//! same keys. Ignored: it asks for a real GPU and minutes (`cargo test
//! sdpa_speed -- --ignored --nocapture`). Each kernel fires `REPS` times a
//! frame; a trial is a frame's wall time over its dispatches, and the best
//! of `TRIALS` is reported.

use std::time::Instant;

use kernels_vulkan::{DecodePlan, KvPool, PrefillPlan, RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

const REPS: usize = 50;
const TRIALS: usize = 7;
const PAGE: usize = 16;
const QH: usize = 32;
const KVH: usize = 8;
const HD: usize = 128;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
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

#[test]
#[ignore = "a benchmark: a real GPU and minutes"]
fn sdpa_speed() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let chip = gpu.device.chip();
    eprintln!("device: {} (subgroup {})", chip.name, chip.subgroup);
    let (profile, _) = engine_vulkan::chip::resolve(&chip, None);
    let tuning = kernels_vulkan::Tuning::new(profile, kernels_vulkan::tuning::Limits::ASSUMED);
    let narrow = chip.subgroup < 32;
    let threads = tuning.sdpa_decode_threads(HD as u32);
    let tile = tuning.sdpa_tile_rows();
    eprintln!("tuning: decode threads {threads}, tile rows {tile}");
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
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

        // Decode: `rows` rows of one request, every row at the last position.
        for rows in [1usize, 16] {
            let q = gpu.up(&noise(3, rows * QH * HD)).0;
            let pos = gpu.up_i32(&vec![kv as i32 - 1; rows]);
            let own = gpu.up_i32(&vec![0; rows]);
            let none = gpu.up_u32(&vec![0; rows.div_ceil(4)]);
            let o = gpu.zeros(rows * QH * HD);
            let (qt, ot) = (
                t(q, rows, QH * HD, Dtype::Bf16),
                t(o, rows, QH * HD, Dtype::Bf16),
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
            let hand = time(&gpu, &|g| {
                g.run(|c| {
                    for _ in 0..REPS {
                        kernels_vulkan::attn::decode(
                            c, &tuning, qt, &plan, &pool, None, HD as u32, scale, ot, None,
                        )?;
                    }
                    Ok(())
                })
            });
            let tier = |subgroup| kernels_slang::attn::sdpa::Tactic {
                tier: if subgroup {
                    kernels_slang::Tier::Subgroup
                } else {
                    kernels_slang::Tier::Baseline
                },
                narrow,
                decode_threads: threads,
                tile_rows: tile,
            };
            let slang = |subgroup: bool| {
                time(&gpu, &|g| {
                    g.run_slang(|c| {
                        for _ in 0..REPS {
                            kernels_slang::attn::sdpa::decode(
                                c,
                                tier(subgroup),
                                st(qt),
                                &tables,
                                &spool,
                                None,
                                HD as u32,
                                scale,
                                st(ot),
                                None,
                            )?;
                        }
                        Ok(())
                    })
                })
            };
            let (sg, base) = (slang(true), slang(false));
            eprintln!(
                "SDPA decode  kv {kv:>5} rows {rows:>3}: hand {hand:>8.1} us  slang subgroup {sg:>8.1} us ({:+.1}%)  slang baseline {base:>8.1} us ({:+.1}%)",
                (sg / hand - 1.0) * 100.0,
                (base / hand - 1.0) * 100.0
            );
        }

        // Prefill: one request's `kv` rows at positions 0..kv.
        let rows = kv.min(2048);
        let q = gpu.up(&noise(4, rows * QH * HD)).0;
        let positions: Vec<i32> = (0..rows as i32).collect();
        let pos = gpu.up_i32(&positions);
        let own = gpu.up_i32(&vec![0; rows]);
        let none = gpu.up_u32(&vec![0; rows.div_ceil(4)]);
        let qi = gpu.up_i32(&[0, rows as i32]);
        let o = gpu.zeros(rows * QH * HD);
        let (qt, ot) = (
            t(q, rows, QH * HD, Dtype::Bf16),
            t(o, rows, QH * HD, Dtype::Bf16),
        );
        let (pt, ow, mt) = (
            t(pos, rows, 1, Dtype::I32),
            t(own, rows, 1, Dtype::I32),
            t(none, rows, 1, Dtype::U8),
        );
        let plan = PrefillPlan {
            positions: pt,
            request_of_token: ow,
            mask: mt,
            mask_enabled: mt,
            mask_stride: 0,
        };
        let rag = RaggedTensor {
            data: qt,
            indptr: t(qi, 2, 1, Dtype::I32),
        };
        let tables = kernels_slang::attn::sdpa::Tables {
            positions: st(pt),
            request_of_token: st(ow),
            mask: st(mt),
            mask_enabled: st(mt),
            mask_stride: 0,
        };
        let hand = time(&gpu, &|g| {
            g.run(|c| {
                for _ in 0..REPS {
                    kernels_vulkan::attn::arbiter::prefill(
                        c, rag, &plan, &pool, None, HD as u32, KVH as u32, scale, ot, 1, &tuning,
                    )?;
                }
                Ok(())
            })
        });
        let tier = kernels_slang::attn::sdpa::Tactic {
            tier: kernels_slang::Tier::Subgroup,
            narrow,
            decode_threads: threads,
            tile_rows: tile,
        };
        let slang = time(&gpu, &|g| {
            g.run_slang(|c| {
                for _ in 0..REPS {
                    kernels_slang::attn::sdpa::prefill(
                        c,
                        tier,
                        st(qt),
                        &tables,
                        &spool,
                        None,
                        HD as u32,
                        scale,
                        st(ot),
                        None,
                    )?;
                }
                Ok(())
            })
        });
        eprintln!(
            "SDPA prefill kv {kv:>5} rows {rows:>4}: hand {hand:>8.1} us  slang tiled {slang:>8.1} us ({:+.1}%)",
            (slang / hand - 1.0) * 100.0
        );
    }
}
