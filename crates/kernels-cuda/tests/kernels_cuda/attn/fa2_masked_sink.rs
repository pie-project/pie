#![cfg(feature = "cuda")]

//! A sinked prefill reading the fire's masks lands, for requests stating
//! none (each word's boundary past every key), the bytes the plain sinked
//! prefill lands: its unsplit rows fold their sinks in the kernel's epilogue
//! as the plain arm's do, not in a pass after.

use crate::common::attn_ref::{col, pool, ten};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, F32, I32, I64, U8, U32};
use kernels_cuda::attn;
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

/// A request's word stating no mask: its boundary past every key.
const NONE: u32 = u32::MAX >> 1;

/// The sinked prefill over `[q heads, kv heads, head dim]` and requests of
/// `(new, cached)` rows, through the masked arm where `masked`: the output
/// and log-sum-exp bytes it lands.
#[allow(clippy::too_many_lines)]
fn fire(
    heads: [usize; 3],
    reqs: &[(usize, usize)],
    window: Option<u32>,
    masked: bool,
) -> (Vec<u16>, Vec<u32>) {
    let ([qh, kvh, d], n) = (heads, reqs.len());
    let (width, qw, scale) = (kvh * d, qh * d, (d as f32).sqrt().recip());
    let (mut qo, mut kv_indptr, mut kv_len, mut last) = (vec![0], vec![0], vec![], vec![]);
    let mut pages: Vec<i32> = Vec::new();
    let mut next = 1usize;
    for (r, &(new, cached)) in reqs.iter().enumerate() {
        let len = new + cached;
        let np = len.div_ceil(PAGE);
        pages.extend((0..np).map(|i| (next + i) as i32));
        next += np;
        qo.push(qo[r] + new as i32);
        kv_indptr.push(kv_indptr[r] + np as i32);
        kv_len.push(len as i32);
        last.push(((len - 1) % PAGE + 1) as i32);
    }
    let (rows, cells) = (qo[n] as usize, next * PAGE);
    let mut lcg = Lcg::seeded(0x51c4 + d as u64);
    let (k, _) = lcg.row(cells * width);
    let (v, _) = lcg.row(cells * width);
    let (q_raw, _) = lcg.row(rows * qw);
    let sinks: Vec<u16> = (0..qh).map(|h| to_bf16(0.5 * (h % 7) as f32 - 1.0)).collect();

    let mut gpu = Gpu::open();
    let ids = col(&mut gpu, &pages, I32);
    let indptr = col(&mut gpu, &kv_indptr, I32);
    let last_page_lens = col(&mut gpu, &last, I32);
    let data = ten(gpu.up(&q_raw), rows, qw, Bf16);
    let qt = RaggedTensor {
        data,
        indptr: col(&mut gpu, &qo, I32),
    };
    let sink_t = col(&mut gpu, &sinks, Bf16);
    // No request states a mask: no bits, every offset 0, every word "none".
    let mask_t = ten(gpu.up(&[0u8; 8]), 8, 1, U8);
    let mask_ip = col(&mut gpu, &vec![0i64; n + 1], I64);
    let origin_t = col(&mut gpu, &vec![NONE; n], U32);
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
    let kvp = KvPool {
        values: ten(gpu.up(&v), cells, width, Bf16),
        last_page_lens,
        head_stride: i64::from(d),
        ..pool(ten(gpu.up(&k), cells, width, Bf16), ids, indptr, PAGE)
    };
    let (o_at, lse_at) = (gpu.zeros(rows * qw * 2), gpu.zeros(rows * qh as usize * 4));
    let (o, lse) = (
        &mut ten(o_at, rows, qw, Bf16),
        &mut ten(lse_at, rows, qh as usize, F32),
    );
    let mut plan = plan::plan_prefill(
        &qo,
        &kv_indptr,
        &kv_len,
        rows as u32,
        shape,
        live,
        window,
        true,
        false,
        masked.then_some(mask_ip),
        &device,
        toggles,
        ws,
    )
    .expect("the prefill plans");
    if masked {
        plan.mask_origin = Some(origin_t);
    }
    plan.stage(&ctx).expect("the plan uploads");
    let mask = masked.then_some(mask_t);
    attn::prefill_lse_sink(&ctx, qt, &plan, &kvp, window, d, kvh, scale, o, lse, sink_t, mask)
        .unwrap_or_else(|e| panic!("{heads:?} {window:?} masked {masked}: {e}"));
    gpu.sync();
    (
        gpu.down::<u16>(o_at, rows * qw),
        gpu.down::<u32>(lse_at, rows * qh as usize),
    )
}

#[test]
fn a_maskless_request_in_the_masked_sinked_prefill_lands_the_plain_bytes() {
    // (new, cached): a prompt alone and chunks over cached rows, past a
    // window and short of it (the golden probe's MiMo script: 37 rows, then
    // 250 over 40).
    let reqs: &[(usize, usize)] = &[(37, 0), (250, 40), (250, 293)];
    for (heads, window) in [([64, 8, 128], Some(128)), ([64, 8, 128], None), ([16, 4, 64], Some(100))] {
        for one in reqs {
            let plain = fire(heads, std::slice::from_ref(one), window, false);
            let masked = fire(heads, std::slice::from_ref(one), window, true);
            assert!(
                plain == masked,
                "{heads:?} {window:?} {one:?}: the masked arm lands other bytes than the plain one"
            );
        }
    }
}
