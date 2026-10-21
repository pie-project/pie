#![cfg(feature = "cuda")]

//! FA2's custom-mask and relative-bias arms over an e4m3 pool read what they
//! read over a bf16 pool holding the same codes: the pages are read as they
//! are, no arm refuses them.

use crate::common::attn_ref::{col, landed, pool, ten};
use crate::common::host::to_e4m3;
use crate::common::tc5::e4m3;
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, E4m3, F32, I32, U8};
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::attn::{self, RelBias, fa2::decode_max_grid_size as grid_of};
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

#[derive(Clone, Copy, Debug)]
enum Arm {
    Masked,
    RelPrefill,
    RelDecode,
}

/// `arm` over `[q heads, kv heads, head dim]` and requests of `(new, cached)`
/// rows, once over an e4m3 pool and once over a bf16 pool of the same codes:
/// the outputs it lands, e4m3 then bf16.
#[allow(clippy::too_many_lines)]
fn both(
    heads: [usize; 3],
    reqs: &[(usize, usize)],
    window: Option<u32>,
    arm: Arm,
) -> [Vec<u16>; 2] {
    let ([qh, kvh, d], n) = (heads, reqs.len());
    let (width, qw, scale) = (kvh * d, qh * d, (d as f32).sqrt().recip());
    let (mut qo, mut kv_indptr, mut kv_len, mut last) = (vec![0], vec![0], vec![], vec![]);
    let mut pages: Vec<i32> = Vec::new();
    let (mut next, mut bits) = (1usize, 0usize);
    let mut mask_indptr = vec![0i32];
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
        mask_indptr.push(bits as i32);
    }
    let (rows, cells) = (qo[n] as usize, next * PAGE);
    let mut lcg = Lcg::seeded(0xf8 + d as u64);
    // The pool's values are e4m3 codes, so both pools hold the same numbers.
    let mut codes = |len: usize| -> (Vec<u8>, Vec<u16>) {
        let (_, x) = lcg.row(len);
        let c: Vec<u8> = x.iter().map(|&v| to_e4m3(2.0 * v)).collect();
        let b = c.iter().map(|&c| to_bf16(e4m3(c))).collect();
        (c, b)
    };
    let ((k8, k16), (v8, v16)) = (codes(cells * width), codes(cells * width));
    let (q_raw, _) = lcg.row(rows * qw);
    // A mask of every bit any reading of its span table could index, set
    // densely enough that each row keeps keys.
    let mask: Vec<u8> = (0..bits.max(8))
        .map(|i| if i % 5 == 0 { 0x0f } else { 0xfb })
        .collect();
    let bias: Vec<f32> = (0..rows * qh * 8)
        .map(|i| 0.25 * ((i % 13) as f32 - 6.0))
        .collect();

    let mut gpu = Gpu::open();
    let ids = col(&mut gpu, &pages, I32);
    let indptr = col(&mut gpu, &kv_indptr, I32);
    let last_page_lens = col(&mut gpu, &last, I32);
    let mask_t = ten(gpu.up(&mask), mask.len(), 1, U8);
    let mask_ip = col(&mut gpu, &mask_indptr, I32);
    let bias_t = ten(gpu.up(&bias), rows, qh * 8, F32);
    let data = ten(gpu.up(&q_raw), rows, qw, Bf16);
    let qt = RaggedTensor {
        data,
        indptr: col(&mut gpu, &qo, I32),
    };
    let (int_ptr, float_ptr, int_bytes) = (gpu.zeros(64 << 20), gpu.zeros(64 << 20), 64 << 20);
    let ws = Workspace {
        int_ptr,
        int_bytes,
        float_ptr,
        float_bytes: int_bytes,
    };
    let ctx = gpu.ctx().with_tuning(Tuning::DEFAULT);
    let device = Device::probe(&ctx).expect("the device answers");
    let toggles = Toggles::of(Tuning::DEFAULT);
    let ([qh, kvh, d], page_size) = ([qh, kvh, d].map(|x| x as u32), PAGE as u32);
    #[rustfmt::skip]
    let shape = Shape { num_requests: n as u32, lane_offset: 0, num_q_heads: qh, num_kv_heads: kvh, head_dim: d, page_size, hnd_layout: false };
    let live = Live {
        requests: n as u32,
        lane_offset: 0,
        row_offset: 0,
        rows: rows as u32,
    };
    let rel = RelBias {
        bias: bias_t,
        extent: 8,
        log_floor: 0,
        log_alpha: 0.0,
    };
    let pools = [
        (
            ten(gpu.up(&k8), cells, width, E4m3),
            ten(gpu.up(&v8), cells, width, E4m3),
        ),
        (
            ten(gpu.up(&k16), cells, width, Bf16),
            ten(gpu.up(&v16), cells, width, Bf16),
        ),
    ];
    pools.map(|(keys, values)| {
        let kvp = KvPool {
            values,
            last_page_lens,
            head_stride: i64::from(d),
            ..pool(keys, ids, indptr, PAGE)
        };
        let o_at = gpu.zeros(rows * qw * 2);
        let o = &mut ten(o_at, rows, qw as usize, Bf16);
        let what = format!("{arm:?} over {qh}/{kvh} heads of {d}, {window:?}");
        let fired = match arm {
            Arm::RelDecode => {
                let grid = grid_of(d, qh, kvh, &device);
                let plan = plan::plan_decode(
                    &kv_indptr, &kv_len, shape, live, window, false, grid, toggles, &device, ws,
                )
                .expect("the decode plans");
                plan.stage(&ctx).expect("the plan uploads");
                attn::decode_rel(&ctx, qt, &plan, &kvp, rel, window, d, scale, o)
            }
            Arm::Masked | Arm::RelPrefill => {
                let masked = matches!(arm, Arm::Masked).then_some(mask_ip);
                let plan = plan::plan_prefill(
                    &qo,
                    &kv_indptr,
                    &kv_len,
                    rows as u32,
                    shape,
                    live,
                    window,
                    true,
                    false,
                    masked,
                    &device,
                    toggles,
                    ws,
                )
                .expect("the prefill plans");
                plan.stage(&ctx).expect("the plan uploads");
                match arm {
                    Arm::Masked => attn::masked(&ctx, qt, &plan, mask_t, &kvp, window, d, scale, o),
                    _ => attn::prefill_rel(&ctx, qt, &plan, &kvp, rel, window, d, kvh, scale, o),
                }
            }
        };
        fired.unwrap_or_else(|e| panic!("{what}: {e}"));
        gpu.sync();
        gpu.down::<u16>(o_at, rows * qw as usize)
    })
}

#[test]
fn the_masked_and_biased_arms_read_an_e4m3_pool() {
    let dec: &[_] = &[(1, 0), (1, 15), (1, 40), (1, 300)];
    let chunks: &[_] = &[(70, 0), (33, 17), (129, 40)];
    let cases: &[([usize; 3], &[(usize, usize)], Option<u32>, Arm)] = &[
        ([16, 4, 128], chunks, None, Arm::Masked),
        ([8, 8, 256], chunks, Some(64), Arm::Masked),
        ([16, 4, 128], chunks, None, Arm::RelPrefill),
        ([8, 2, 64], chunks, Some(48), Arm::RelPrefill),
        ([16, 4, 128], dec, None, Arm::RelDecode),
        ([8, 2, 64], dec, Some(48), Arm::RelDecode),
    ];
    for &(heads, reqs, window, arm) in cases {
        let [fp8, bf16] = both(heads, reqs, window, arm);
        let what = format!("{arm:?} over {heads:?}, {window:?}");
        let want: Vec<f32> = bf16.iter().map(|&b| crate::common::from_bf16(b)).collect();
        let mut gpu = Gpu::open();
        let at = gpu.up(&fp8);
        landed(&gpu, at, &want, 2e-2, 0.5, &what);
    }
}
