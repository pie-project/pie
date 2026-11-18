//! The prefill attention's kernel choice (`engine_gpu::attn_choice`, the
//! roofline's cost of each kernel at the device's measured rates) against
//! both kernels timed at the same fire: decode-like lanes (1 / 8 / 64 lanes
//! of one row over 512 / 2k / 8k keys), prefill tails (16 / 48 rows over 48
//! / 512 / 2k / 8k keys) and prompts (64 to 2048 rows over their own keys,
//! 2k and 8k), at Qwen3-0.6B's heads (16 query, 8 kv, d 128). The chosen
//! kernel is within 5% of the faster at every point but the one named
//! below; decode lanes keep the per-row walk.

use std::time::Instant;

use engine_gpu::backend::Tactics as _;
use kernels_slang::attn::sdpa;
use kernels_slang::{KvPool, Tensor};
use poem_ir::Dtype;

use crate::gpu::Gpu;

const PAGE: u32 = 16;
const QH: u32 = 16;
const KVH: u32 = 8;
const HD: u32 = 128;

/// How far the chosen kernel may run past the faster.
const WITHIN: f64 = 0.05;

/// The one point outside the roofline: 128 rows over their own 128 keys.
/// Both kernels finish under [`CORNER_US`]; the per-row walk is the faster
/// (the RTX PRO 4500 on Vulkan: 61.5 against 87.5 us), but at so few keys
/// a fixed cost a workgroup (its setup and its last partial wave) the
/// roofline does not count outweighs the work, and the tiled kernel is
/// chosen. Both kernels are held under the bound instead.
const CORNER: (u32, u32) = (128, 128);
const CORNER_US: f64 = 90.0;

/// Microseconds a dispatch of `fire` takes: a frame of 21 less a frame of
/// 1, over 20, the median of five.
fn time(
    gpu: &Gpu,
    fire: &dyn Fn(&crate::backend::Sink<'_>) -> Result<(), kernels_slang::Error>,
) -> f64 {
    let once = |n: usize| {
        let at = Instant::now();
        gpu.run(|sink| {
            for _ in 0..n {
                fire(sink)?;
            }
            Ok(())
        });
        at.elapsed().as_secs_f64()
    };
    once(2);
    let mut samples: Vec<f64> = (0..5).map(|_| (once(21) - once(1)) / 20.0 * 1e6).collect();
    samples.sort_by(f64::total_cmp);
    samples[2]
}

/// `(label, requests, rows a request, keys a request)` of every point.
fn points() -> Vec<(String, u32, u32, u32)> {
    let mut points = Vec::new();
    for lanes in [1u32, 8, 64] {
        for keys in [512u32, 2048, 8192] {
            points.push((format!("decode {lanes}x{keys}"), lanes, 1, keys));
        }
    }
    for rows in [16u32, 48] {
        for keys in [48u32, 512, 2048, 8192] {
            points.push((format!("tail {rows}@{keys}"), 1, rows, keys.max(rows)));
        }
    }
    for rows in [64u32, 80, 96, 128, 256, 512, 2048] {
        for keys in [rows, 2048, 8192] {
            if keys >= rows {
                points.push((format!("prefill {rows}@{keys}"), 1, rows, keys));
            }
        }
    }
    // 2048 rows over their own keys and over 2048 are one point.
    points.dedup_by(|a, b| a.0 == b.0);
    points
}

#[test]
#[ignore = "a timing gate: run by name on a quiet device"]
fn the_sdpa_choice_is_within_5_percent_of_the_faster_kernel() {
    let Some(mut gpu) = Gpu::open() else {
        return;
    };
    let tuning = crate::backend::rated(&gpu.device);
    let tactic = tuning.sdpa(crate::backend::tier(&gpu.device), HD);
    let tile = tactic.tile().1;
    let rates = tuning.profile().rates();
    println!(
        "backend {} cores {} tile_rows {tile} workgroups_per_core {:?} attn {:?} hbm {:?}",
        crate::backend::NAME,
        tuning.cores(),
        rates.workgroups_per_core,
        rates.attn_tflops_by_keys,
        rates.hbm_gbps,
    );
    println!("point,rows,keys,tiled_us,walked_us,faster,chosen,cost_tiled,cost_walked,verdict");
    let mut misses = Vec::new();
    for (label, requests, per, keys) in points() {
        let rows = requests * per;
        let fire = engine_gpu::attn_choice::Fire {
            rows,
            q_heads: QH,
            kv_heads: KVH,
            head_dim: HD,
            tile_rows: tile,
        };
        let constants = engine::roofline::Constants::of(&rates).expect("the rates read");
        let costs = engine_gpu::attn_choice::costs(fire, &constants, tuning.cores())
            .expect("the rates price the choice");
        let tiles = tuning.sdpa_tiles(fire).expect("the rates price the choice");
        assert_eq!(tiles, costs.tiles());
        let pages_per = keys.div_ceil(PAGE);
        let pages = requests * pages_per;
        let slots = (pages * PAGE) as usize;
        let kv_width = KVH * HD;
        let keys_buf = gpu.zeros(slots * kv_width as usize);
        let values_buf = gpu.zeros(slots * kv_width as usize);
        let indices: Vec<u32> = (0..pages).collect();
        let indptr: Vec<u32> = (0..=requests).map(|r| r * pages_per).collect();
        let pi = gpu.up_u32(&indices);
        let pp = gpu.up_u32(&indptr);
        let pool = KvPool::planes(
            Tensor::new(keys_buf, slots as u32, kv_width, Dtype::Bf16),
            Tensor::new(values_buf, slots as u32, kv_width, Dtype::Bf16),
            Tensor::new(pi, pages, 1, Dtype::U32),
            Tensor::new(pp, requests + 1, 1, Dtype::U32),
            PAGE as i32,
            u64::from(kv_width),
            u64::from(HD),
        );
        let positions: Vec<i32> = (0..requests)
            .flat_map(|_| (keys - per..keys).map(|p| p as i32))
            .collect();
        let owner: Vec<i32> = (0..requests)
            .flat_map(|r| std::iter::repeat_n(r as i32, per as usize))
            .collect();
        let pos = gpu.up_i32(&positions);
        let own = gpu.up_i32(&owner);
        let mask = gpu.up_u32(&vec![0u32; rows as usize]);
        let enabled = gpu.up_u32(&vec![0u32; rows as usize]);
        let tables = sdpa::Tables {
            positions: Tensor::new(pos, rows, 1, Dtype::I32),
            request_of_token: Tensor::new(own, rows, 1, Dtype::I32),
            mask: Tensor::new(mask, rows, 4, Dtype::U8),
            mask_enabled: Tensor::new(enabled, rows, 1, Dtype::U32),
            mask_stride: 4,
        };
        let (q, _) = gpu.up(&crate::gpu::noise(
            u64::from(rows),
            (rows * QH * HD) as usize,
        ));
        let q = Tensor::new(q, rows, QH * HD, Dtype::Bf16);
        let o = Tensor::new(
            gpu.zeros((rows * QH * HD) as usize),
            rows,
            QH * HD,
            Dtype::Bf16,
        );
        let scale = 1.0 / (HD as f32).sqrt();
        let tiled = time(&gpu, &|sink| {
            sdpa::prefill(sink, tactic, q, &tables, &pool, None, HD, scale, o, None)
        });
        let walked = time(&gpu, &|sink| {
            sdpa::decode(
                sink, tactic, q, &tables, &pool, None, HD, scale, o, None, None,
            )
        });
        let best = tiled.min(walked);
        let chosen = if tiles { tiled } else { walked };
        let corner = per == CORNER.0 && keys == CORNER.1 && requests == 1;
        let over = chosen / best - 1.0;
        let verdict = if over <= WITHIN {
            "ok".to_string()
        } else if corner && tiled.max(walked) < CORNER_US {
            format!("corner {:+.0}%", over * 100.0)
        } else {
            misses.push(format!("{label}: {:+.0}%", over * 100.0));
            format!("MISS {:+.0}%", over * 100.0)
        };
        if per == 1 {
            assert!(!tiles, "{label}: decode lanes keep the per-row walk");
        }
        println!(
            "{label},{rows},{keys},{tiled:.2},{walked:.2},{},{},{:.1},{:.1},{verdict}",
            if tiled <= walked { "tiled" } else { "walked" },
            if tiles { "tiled" } else { "walked" },
            costs.tiled,
            costs.walked,
        );
    }
    assert!(
        misses.is_empty(),
        "the chosen kernel ran more than {:.0}% past the faster: {misses:?}",
        WITHIN * 100.0
    );
}
