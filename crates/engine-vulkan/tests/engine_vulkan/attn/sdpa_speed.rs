//! Paged attention's speed on the first Vulkan device: the hand-written
//! decode (plain, split into key ranges, selected), its few-rows prefill arm
//! and the tiled prefill beside kernels-slang's, the same shapes and the
//! same keys. Ignored: it asks for a real GPU and minutes (`cargo test
//! sdpa_speed -- --ignored --nocapture`). Each kernel fires `REPS` times a
//! frame; a trial is the frame's span on the device's timestamps over its
//! repetitions, the kernels of a shape take their trials in turn, and the
//! best of `TRIALS` is reported.

use kernels_slang::attn::sdpa;
use kernels_vulkan::{DecodePlan, KvPool, PrefillPlan, RaggedTensor, Split, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

const REPS: usize = 100;
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

/// The best trial's microseconds a repetition of each runner on the
/// device's clock (the span of the frame's timestamps), the runners taking
/// their trials in turn.
fn race(gpu: &Gpu, runs: &[Run<'_>]) -> Vec<f64> {
    for run in runs {
        run(gpu);
    }
    let mut best = vec![f64::INFINITY; runs.len()];
    for _ in 0..TRIALS {
        for (run, best) in runs.iter().zip(&mut best) {
            engine_vulkan::encode::reset_kernel_profile();
            run(gpu);
            let span = engine_vulkan::encode::frame_span_ns() as f64 / 1e3;
            *best = best.min(span / REPS as f64);
        }
    }
    best
}

fn report(shape: &str, times: &[f64], names: &[&str]) {
    let hand = times[0];
    let mut line = format!("SDPA {shape:<34} hand {hand:>8.1} us");
    for (t, name) in times[1..].iter().zip(names) {
        line += &format!("  {name} {t:>8.1} us ({:+.1}%)", (t / hand - 1.0) * 100.0);
    }
    eprintln!("{line}");
}

#[test]
#[ignore = "a benchmark: a real GPU and minutes"]
fn sdpa_speed() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    // Every frame is timed on the device's clock.
    engine_vulkan::encode::profile_kernels(true);
    let chip = gpu.device.chip();
    eprintln!("device: {} (subgroup {})", chip.name, chip.subgroup);
    let (profile, _) = engine_vulkan::chip::resolve(&chip, None);
    let tuning = &kernels_vulkan::Tuning::new(profile, gpu.device.limits());
    let tactic = |tier| sdpa::Tactic {
        tier,
        narrow: tuning.limits().lanes < 32,
        decode_threads: tuning.sdpa_decode_threads(HD as u32),
        tile_rows: tuning.sdpa_tile_rows(),
        split_groups: tuning.sdpa_split_groups(),
        split_min_keys: tuning.sdpa_split_min_keys(),
    };
    let (subgroup, baseline) = (
        tactic(kernels_slang::Tier::Subgroup),
        tactic(kernels_slang::Tier::Baseline),
    );
    eprintln!("tactic: {subgroup:?}");
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
            let part_out = gpu.up_f32(&vec![0.0; SPLITS * rows * QH * HD]);
            let part_lse = gpu.up_f32(&vec![0.0; SPLITS * rows * QH]);
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
                part_out: t(part_out, SPLITS * rows * QH, HD, Dtype::F32),
                part_lse: t(part_lse, SPLITS * rows, QH, Dtype::F32),
            };
            let ssplit = sdpa::Split {
                part_out: st(split.part_out),
                part_lse: st(split.part_lse),
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
                            kernels_vulkan::attn::decode(
                                c, tuning, qt, plan, pool, None, HD as u32, scale, ot, split,
                            )?;
                        }
                        Ok(())
                    });
                }
            };
            let slang = |tactic: sdpa::Tactic, split: Option<sdpa::Split>| {
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
            let names = ["slang subgroup", "slang baseline"];
            let times = race(
                &gpu,
                &[&hand(None), &slang(subgroup, None), &slang(baseline, None)],
            );
            report(&format!("decode kv {kv} rows {rows}"), &times, &names);
            let times = race(
                &gpu,
                &[
                    &hand(Some(split)),
                    &slang(subgroup, Some(ssplit)),
                    &slang(baseline, Some(ssplit)),
                ],
            );
            report(&format!("split decode kv {kv} rows {rows}"), &times, &names);

            let few = |g: &Gpu| {
                g.run(|c| {
                    for _ in 0..REPS {
                        kernels_vulkan::attn::arbiter::prefill(
                            c, rag, &pplan, pool, None, HD as u32, KVH as u32, scale, ot, 1, tuning,
                        )?;
                    }
                    Ok(())
                });
            };
            if (rows as u32) < tuning.sdpa_tile_rows_per_request() {
                let times = race(&gpu, &[&few, &slang(subgroup, None)]);
                report(
                    &format!("few-rows prefill kv {kv} rows {rows}"),
                    &times,
                    &names[..1],
                );
            }

            if kv >= RATIO * (TOP_K + 1) {
                let hand = |g: &Gpu| {
                    g.run(|c| {
                        for _ in 0..REPS {
                            kernels_vulkan::attn::decode_selected(
                                c,
                                tuning,
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
                let slang = |tactic: sdpa::Tactic| {
                    move |g: &Gpu| {
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
                    }
                };
                let times = race(&gpu, &[&hand, &slang(subgroup), &slang(baseline)]);
                report(&format!("selected kv {kv} rows {rows}"), &times, &names);
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
                for _ in 0..REPS {
                    kernels_vulkan::attn::arbiter::prefill(
                        c, rag, plan, pool, None, HD as u32, KVH as u32, scale, ot, 1, tuning,
                    )?;
                }
                Ok(())
            });
        };
        let slang = |g: &Gpu| {
            g.run_slang(|c| {
                for _ in 0..REPS {
                    sdpa::prefill(
                        c,
                        subgroup,
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
        };
        let times = race(&gpu, &[&hand, &slang]);
        report(
            &format!("prefill kv {kv} rows {rows}"),
            &times,
            &["slang tiled"],
        );
    }
}
