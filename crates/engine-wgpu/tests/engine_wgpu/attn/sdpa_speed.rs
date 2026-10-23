//! Paged attention's speed on the first wgpu adapter: kernels-wgpu's decode
//! (plain, split into key ranges, selected), its few-rows prefill arm and
//! its tiled prefill beside kernels-slang's at the tactic engine-wgpu runs,
//! the same shapes and the same keys. Ignored: it asks for a real GPU and
//! minutes (`cargo test sdpa_speed -- --ignored --nocapture`). Each kernel
//! fires `reps` times a frame; a trial is the dispatches' time on the
//! device's timestamps (the wall clock where the adapter has none) over its
//! repetitions, the kernels of a shape take their trials in turn, and the
//! best of `TRIALS` is reported.

use kernels_slang::attn::sdpa;
use kernels_wgpu::attn::Split;
use kernels_wgpu::{DecodePlan, KvPool, PrefillPlan, RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

const REPS: usize = 100;
/// Repetitions a frame of a tiled prefill, each milliseconds long.
const PREFILL_REPS: usize = 10;
const TRIALS: usize = 15;
const PAGE: usize = 16;
const QH: usize = 32;
const KVH: usize = 8;
const HD: usize = 128;
/// The selected form's block width and blocks a row: 2048 keys.
const RATIO: usize = 64;
const TOP_K: usize = 32;
/// The partial planes hold this many splits.
const SPLITS: usize = 32;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

type Run<'a> = &'a dyn Fn(&Gpu);

/// The device's nanoseconds over the frame's dispatches, or the wall
/// clock's over the whole run where the adapter writes no timestamps.
fn span_ns(gpu: &Gpu, run: Run<'_>) -> f64 {
    engine_wgpu::encode::reset_kernel_profile();
    let wall = std::time::Instant::now();
    run(gpu);
    let wall = wall.elapsed().as_nanos() as f64;
    let device: u64 = engine_wgpu::encode::kernel_profile()
        .iter()
        .filter(|(name, ..)| name != "frame.span")
        .map(|(_, ns, _)| ns)
        .sum();
    if device == 0 { wall } else { device as f64 }
}

/// The best trial's microseconds a repetition of each runner, the runners
/// taking their trials in turn.
fn race(gpu: &Gpu, reps: usize, runs: &[Run<'_>]) -> Vec<f64> {
    for run in runs {
        run(gpu);
    }
    let mut best = vec![f64::INFINITY; runs.len()];
    for _ in 0..TRIALS {
        for (run, best) in runs.iter().zip(&mut best) {
            *best = best.min(span_ns(gpu, *run) / 1e3 / reps as f64);
        }
    }
    best
}

fn report(shape: &str, times: &[f64]) {
    let (hand, slang) = (times[0], times[1]);
    eprintln!(
        "SDPA {shape:<34} hand {hand:>8.1} us  slang {slang:>8.1} us ({:+.1}%)",
        (slang / hand - 1.0) * 100.0
    );
}

#[test]
#[ignore = "a benchmark: a real GPU and minutes"]
fn sdpa_speed() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    engine_wgpu::encode::profile_timing(true);
    let chip = gpu.device.chip();
    eprintln!("adapter: {}", chip.name);
    let (profile, _) = engine_wgpu::chip::resolve(&chip, None);
    let tuning =
        &kernels_wgpu::Tuning::new(profile, gpu.device.cores(), gpu.device.storage_buffers());
    // engine-wgpu's tactic (`dispatch/attn.rs`).
    let tactic = sdpa::Tactic {
        tier: if chip.subgroups_32() {
            kernels_slang::Tier::Subgroup
        } else {
            kernels_slang::Tier::Baseline
        },
        narrow: false,
        decode_threads: 256,
        tile_rows: 16,
        tile_lanes: 16,
        split_groups: tuning.sdpa_split_groups(),
        split_min_keys: tuning.sdpa_split_min_keys(),
    };
    eprintln!("tactic: {tactic:?}");
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let scale = 1.0 / (HD as f32).sqrt();

    for kv in [1024usize, 8192] {
        let pages = kv.div_ceil(PAGE);
        let slots = pages * PAGE;
        let page_ids: Vec<u32> = (0..pages as u32).rev().collect();
        let (pi, pp) = (gpu.up_u32(&page_ids), gpu.up_u32(&[0, pages as u32]));
        let keys = gpu.up(&noise(1, slots * KVH * HD)).0;
        let values = gpu.up(&noise(2, slots * KVH * HD)).0;
        let pool = &KvPool {
            keys: t(keys, slots, KVH * HD, Dtype::Bf16),
            values: t(values, slots, KVH * HD, Dtype::Bf16),
            page_indices: t(pi, pages, 1, Dtype::U32),
            page_indptr: t(pp, 2, 1, Dtype::U32),
            page_size: PAGE as i32,
            seq_stride: (KVH * HD) as u64,
            head_stride: HD as u64,
        };
        let spool = &kernels_slang::KvPool {
            keys: st(pool.keys),
            values: st(pool.values),
            page_indices: st(pool.page_indices),
            page_indptr: st(pool.page_indptr),
            page_size: pool.page_size,
            seq_stride: pool.seq_stride,
            head_stride: pool.head_stride,
        };

        // Decode: `rows` rows of one request at its last positions.
        for rows in [1usize, 4, 16] {
            let q = gpu.up(&noise(3, rows * QH * HD)).0;
            let positions: Vec<i32> = (0..rows).map(|r| (kv - rows + r) as i32).collect();
            let pos = gpu.up_i32(&positions);
            let own = gpu.up_i32(&vec![0; rows]);
            let none = gpu.up_u32(&vec![0; rows.div_ceil(4)]);
            let qi = gpu.up_i32(&[0, rows as i32]);
            let blocks: Vec<i32> = (0..rows * TOP_K)
                .map(|i| ((i * 7) % (kv / RATIO - 1)) as i32)
                .collect();
            let sel = gpu.up_i32(&blocks);
            let o = gpu.zeros(rows * QH * HD);
            let (qt, ot, selt) = (
                t(q, rows, QH * HD, Dtype::Bf16),
                t(o, rows, QH * HD, Dtype::Bf16),
                t(sel, rows, TOP_K, Dtype::I32),
            );
            let (pt, ow, mt) = (
                t(pos, rows, 1, Dtype::I32),
                t(own, rows, 1, Dtype::I32),
                t(none, rows, 1, Dtype::U8),
            );
            let split = Split {
                o: t(
                    gpu.zeros(SPLITS * rows * QH * HD),
                    SPLITS * rows,
                    QH * HD,
                    Dtype::Bf16,
                ),
                lse: t(
                    gpu.up_f32(&vec![0.0; SPLITS * rows * QH]),
                    SPLITS * rows,
                    QH,
                    Dtype::F32,
                ),
                keys: kv as u32,
            };
            let ssplit = sdpa::Split {
                part_out: st(t(
                    gpu.up_f32(&vec![0.0; SPLITS * rows * QH * HD]),
                    SPLITS * rows * QH,
                    HD,
                    Dtype::F32,
                )),
                part_lse: st(t(
                    gpu.up_f32(&vec![0.0; SPLITS * rows * QH]),
                    SPLITS * rows,
                    QH,
                    Dtype::F32,
                )),
            };
            let plan = &DecodePlan {
                positions: pt,
                request_of_token: ow,
                mask: mt,
                mask_enabled: mt,
                mask_stride: 0,
            };
            let pplan = PrefillPlan {
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
            let tables = &sdpa::Tables {
                positions: st(pt),
                request_of_token: st(ow),
                mask: st(mt),
                mask_enabled: st(mt),
                mask_stride: 0,
            };
            let hand = |split: Option<Split>| {
                move |g: &Gpu| {
                    g.run(|c| {
                        for _ in 0..REPS {
                            kernels_wgpu::attn::decode(
                                c, tuning, qt, plan, pool, None, HD as u32, scale, ot, split,
                            )?;
                        }
                        Ok(())
                    });
                }
            };
            let slang = |split: Option<sdpa::Split>| {
                move |g: &Gpu| {
                    g.run_slang(|c| {
                        for _ in 0..REPS {
                            sdpa::decode(
                                c,
                                tactic,
                                st(qt),
                                tables,
                                spool,
                                None,
                                HD as u32,
                                scale,
                                st(ot),
                                None,
                                split,
                            )?;
                        }
                        Ok(())
                    });
                }
            };
            let times = race(&gpu, REPS, &[&hand(None), &slang(None)]);
            report(&format!("decode kv {kv} rows {rows}"), &times);
            let times = race(&gpu, REPS, &[&hand(Some(split)), &slang(Some(ssplit))]);
            report(&format!("split decode kv {kv} rows {rows}"), &times);

            let few = |g: &Gpu| {
                g.run(|c| {
                    for _ in 0..REPS {
                        kernels_wgpu::attn::arbiter::prefill(
                            c, rag, &pplan, pool, None, HD as u32, KVH as u32, scale, ot, 1, tuning,
                        )?;
                    }
                    Ok(())
                });
            };
            if (rows as u32) < tuning.sdpa_tile_rows_per_request() {
                let times = race(&gpu, REPS, &[&few, &slang(None)]);
                report(&format!("few-rows prefill kv {kv} rows {rows}"), &times);
            }

            if kv >= RATIO * (TOP_K + 1) {
                let hand = |g: &Gpu| {
                    g.run(|c| {
                        for _ in 0..REPS {
                            kernels_wgpu::attn::decode_selected(
                                c,
                                qt,
                                plan,
                                selt,
                                pool,
                                None,
                                HD as u32,
                                scale,
                                RATIO as u32,
                                ot,
                            )?;
                        }
                        Ok(())
                    });
                };
                let slang = |g: &Gpu| {
                    g.run_slang(|c| {
                        for _ in 0..REPS {
                            sdpa::selected(
                                c,
                                tactic,
                                st(qt),
                                tables,
                                spool,
                                HD as u32,
                                scale,
                                RATIO as u32,
                                st(selt),
                                st(ot),
                            )?;
                        }
                        Ok(())
                    });
                };
                let times = race(&gpu, REPS, &[&hand, &slang]);
                report(&format!("selected kv {kv} rows {rows}"), &times);
            }
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
        let plan = &PrefillPlan {
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
        let tables = &sdpa::Tables {
            positions: st(pt),
            request_of_token: st(ow),
            mask: st(mt),
            mask_enabled: st(mt),
            mask_stride: 0,
        };
        let hand = |g: &Gpu| {
            g.run(|c| {
                for _ in 0..PREFILL_REPS {
                    kernels_wgpu::attn::arbiter::prefill(
                        c, rag, plan, pool, None, HD as u32, KVH as u32, scale, ot, 1, tuning,
                    )?;
                }
                Ok(())
            });
        };
        let slang = |tile_rows: u32, tile_lanes: u32, tier: kernels_slang::Tier| {
            move |g: &Gpu| {
                g.run_slang(|c| {
                    for _ in 0..PREFILL_REPS {
                        sdpa::prefill(
                            c,
                            sdpa::Tactic {
                                tier,
                                tile_rows,
                                tile_lanes,
                                ..tactic
                            },
                            st(qt),
                            tables,
                            spool,
                            None,
                            HD as u32,
                            scale,
                            st(ot),
                            None,
                        )?;
                    }
                    Ok(())
                });
            }
        };
        // Every tile the adapter's workgroup holds, at the baseline and
        // (the 16-lane tile) at the tactic's tier.
        let invocations = gpu.device.workgroup_limits().0;
        let mut tiles = vec![(8u32, 32u32, kernels_slang::Tier::Baseline)];
        tiles.extend(
            [(16, 32), (32, 32), (16, 16)].map(|(t, l)| (t, l, kernels_slang::Tier::Baseline)),
        );
        if tactic.tier != kernels_slang::Tier::Baseline {
            tiles.push((16, 16, tactic.tier));
        }
        for (tile, lanes, tier) in tiles {
            if lanes * tile <= invocations {
                let times = race(&gpu, PREFILL_REPS, &[&hand, &slang(tile, lanes, tier)]);
                report(
                    &format!(
                        "prefill kv {kv} rows {rows} (tiled {tile}x{lanes} {})",
                        tier.name()
                    ),
                    &times,
                );
            }
        }
    }
}
