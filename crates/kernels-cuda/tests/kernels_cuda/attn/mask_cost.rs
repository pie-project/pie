#![cfg(feature = "cuda")]

//! What a lane's custom mask costs: a decode lane's masked path (FA2's
//! prefill with a mask, a row a lane) against the decode kernel, and FA2's
//! prefill masked arm over an all-admit mask against its maskless arm. A
//! probe run by hand (`--ignored --nocapture`), timed over many launches.

use crate::common::attn_ref::{col, pool, ten};
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, I32, I64, U8};
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::attn::{self, fa2::decode_max_grid_size as grid_of};
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;
const REPS: usize = 200;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    Decode,
    DecodeBody,
    MaskedDecode,
    Prefill,
    MaskedPrefill,
}

/// Microseconds a launch of `arm` over `[q heads, kv heads, dim]` and
/// requests of `(new, cached)` rows takes, every mask bit admitting.
fn time(heads: [usize; 3], reqs: &[(usize, usize)], arm: Arm) -> f64 {
    let ([qh, kvh, d], n) = (heads, reqs.len());
    let (width, qw, scale) = (kvh * d, qh * d, (d as f32).sqrt().recip());
    let (mut qo, mut kv_indptr, mut kv_len, mut last) = (vec![0], vec![0], vec![], vec![]);
    let mut pages: Vec<i32> = Vec::new();
    let (mut next, mut bits) = (1usize, 0usize);
    let mut mask_indptr = vec![0i64];
    for (r, &(new, prefix)) in reqs.iter().enumerate() {
        let len = new + prefix;
        let np = len.div_ceil(PAGE);
        pages.extend((0..np).map(|i| (next + i) as i32));
        next += np;
        qo.push(qo[r] + new as i32);
        kv_indptr.push(kv_indptr[r] + np as i32);
        kv_len.push(len as i32);
        last.push(((len - 1) % PAGE + 1) as i32);
        bits += new * len;
        mask_indptr.push(bits as i64);
    }
    let (rows, cells) = (qo[n] as usize, next * PAGE);
    let mut lcg = Lcg::seeded(0x5ca1e + d as u64);
    let (k, _) = lcg.row(cells * width);
    let (v, _) = lcg.row(cells * width);
    let (q_raw, _) = lcg.row(rows * qw);
    let mask = vec![0xffu8; bits.div_ceil(8).max(8)];

    let mut gpu = Gpu::open();
    let ids = col(&mut gpu, &pages, I32);
    let indptr = col(&mut gpu, &kv_indptr, I32);
    let last_page_lens = col(&mut gpu, &last, I32);
    let mask_t = ten(gpu.up(&mask), mask.len(), 1, U8);
    let mask_ip = col(&mut gpu, &mask_indptr, I64);
    let qt = RaggedTensor {
        data: ten(gpu.up(&q_raw), rows, qw, Bf16),
        indptr: col(&mut gpu, &qo, I32),
    };
    let (int_ptr, float_ptr, int_bytes) = (gpu.zeros(64 << 20), gpu.zeros(64 << 20), 64 << 20);
    let ws = Workspace {
        int_ptr,
        int_bytes,
        float_ptr,
        float_bytes: int_bytes,
    };
    // The L40S profile's attention decisions (`profiles/sm89/nvidia-l40s.toml`);
    // the rest are the registry's defaults, as the profile leaves them.
    let tuning = Tuning {
        merge_warp: true,
        fa2_live: true,
        ..Tuning::DEFAULT
    };
    let ctx = gpu.ctx().with_tuning(tuning);
    let device = Device::probe(&ctx).expect("the device answers");
    let toggles = Toggles::of(tuning);
    let ([qh, kvh, d], page_size) = ([qh, kvh, d].map(|x| x as u32), PAGE as u32);
    #[rustfmt::skip]
    let shape = Shape { num_requests: n as u32, lane_offset: 0, num_q_heads: qh, num_kv_heads: kvh, head_dim: d, page_size, hnd_layout: false };
    let live = Live {
        requests: n as u32,
        lane_offset: 0,
        row_offset: 0,
        rows: rows as u32,
    };
    let kvp = KvPool {
        values: ten(gpu.up(&v), cells, width, Bf16),
        last_page_lens,
        head_stride: i64::from(d),
        ..pool(ten(gpu.up(&k), cells, width, Bf16), ids, indptr, PAGE)
    };
    let o = &mut ten(gpu.zeros(rows * qw * 2), rows, qw, Bf16);
    let mut launch: Box<dyn FnMut()> = match arm {
        Arm::Decode | Arm::DecodeBody => {
            let grid = grid_of(d, qh, kvh, &device);
            let body = arm == Arm::DecodeBody;
            let planned = std::time::Instant::now();
            for _ in 0..REPS {
                std::hint::black_box(
                    plan::plan_decode(
                        &kv_indptr, &kv_len, shape, live, None, body, grid, toggles, &device, ws,
                    )
                    .expect("the decode plans"),
                );
            }
            println!(
                "mask-cost plan host {arm:?} lanes={n} keys={}: {:.2} us",
                kv_len[0],
                planned.elapsed().as_secs_f64() * 1e6 / REPS as f64
            );
            let plan = plan::plan_decode(
                &kv_indptr, &kv_len, shape, live, None, body, grid, toggles, &device, ws,
            )
            .expect("the decode plans");
            plan.stage(&ctx).expect("the plan uploads");
            Box::new(move || attn::decode(&ctx, qt, &plan, &kvp, None, d, scale, o, None).unwrap())
        }
        _ => {
            let masked = matches!(arm, Arm::MaskedDecode | Arm::MaskedPrefill).then_some(mask_ip);
            let plan = plan::plan_prefill(
                &qo,
                &kv_indptr,
                &kv_len,
                rows as u32,
                shape,
                live,
                None,
                true,
                false,
                masked,
                &device,
                toggles,
                ws,
            )
            .expect("the prefill plans");
            plan.stage(&ctx).expect("the plan uploads");
            if masked.is_some() {
                Box::new(move || attn::masked(&ctx, qt, &plan, mask_t, &kvp, None, d, scale, o).unwrap())
            } else {
                Box::new(move || attn::prefill(&ctx, qt, &plan, &kvp, None, d, kvh, scale, o).unwrap())
            }
        }
    };
    for _ in 0..10 {
        launch();
    }
    gpu.sync();
    let began = std::time::Instant::now();
    for _ in 0..REPS {
        launch();
    }
    gpu.sync();
    began.elapsed().as_secs_f64() * 1e6 / REPS as f64
}

#[test]
#[ignore = "a probe run by hand"]
fn what_a_lanes_mask_costs() {
    let shapes = [("qwen3-0.6b", [16, 8, 128]), ("qwen3-8b", [32, 8, 128])];
    for (name, heads) in shapes {
        for lanes in [1usize, 8, 64, 256] {
            for keys in [512usize, 4096] {
                let reqs: Vec<(usize, usize)> = vec![(1, keys - 1); lanes];
                let dec = time(heads, &reqs, Arm::Decode);
                let body = time(heads, &reqs, Arm::DecodeBody);
                println!("mask-cost {name} decode lanes={lanes} keys={keys}: eager plan {dec:.1} us, body plan {body:.1} us");
                let masked = time(heads, &reqs, Arm::MaskedDecode);
                let pre = time(heads, &reqs, Arm::Prefill);
                println!(
                    "mask-cost {name} decode lanes={lanes} keys={keys}: decode {dec:.1} us, \
                     maskless prefill a row a lane {pre:.1} us, masked prefill a row a lane {masked:.1} us ({:.2}x decode)",
                    masked / dec
                );
            }
        }
        for rows in [512usize, 2048] {
            let reqs = [(rows, 0usize)];
            let plain = time(heads, &reqs, Arm::Prefill);
            let masked = time(heads, &reqs, Arm::MaskedPrefill);
            println!(
                "mask-cost {name} prefill rows={rows}: maskless {plain:.1} us, masked all-admit {masked:.1} us ({:.2}x)",
                masked / plain
            );
        }
    }
}
