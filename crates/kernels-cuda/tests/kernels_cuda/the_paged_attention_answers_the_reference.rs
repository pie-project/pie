#![cfg(feature = "cuda")]

//! The paged attention launchers (`decode_lse`, FA2's and tcgen05's
//! `prefill_lse` and their sinked forms) against the host softmax, one case a route.

use crate::common::attn_ref::{attend, col, landed, near, pool, ten, tuned};
use crate::common::{Gpu, Lcg, from_bf16, host::upload_to};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::attn::{self, fa2::decode_max_grid_size as grid_of};
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

/// A decode read: the warp merge, the tensor-core decode, its whole window
/// and its cluster.
fn read(warp: bool, xqa: bool, whole: u32, cluster: u32) -> Tuning {
    tuned(|t| {
        (t.merge_warp, t.decode_xqa) = (warp, xqa);
        (t.decode_xqa_whole_window, t.decode_xqa_cluster) = (whole, cluster);
    })
}

/// The decode over the lanes `(first, count)` of a fire (every lane where
/// `Some(None)`), or the prefill where `None`.
type Lanes = Option<Option<(u32, u32)>>;

/// `[q heads, kv heads, head dim]` over requests of `(new, cached)` rows.
fn check(
    heads: [usize; 3],
    reqs: &[(usize, usize)],
    window: Option<u32>,
    flags: [bool; 2],
    lanes: Lanes,
    tuning: Tuning,
) {
    check_live(heads, reqs, window, flags, lanes, tuning, [0, 0]);
}

/// [`check`] with heads zero-padded past `live` (a key's and a value's
/// widths; `[0, 0]` none), the pool saying so.
#[allow(clippy::too_many_lines)]
fn check_live(
    heads: [usize; 3],
    reqs: &[(usize, usize)],
    window: Option<u32>,
    flags: [bool; 2],
    lanes: Lanes,
    tuning: Tuning,
    live: [usize; 2],
) {
    let ([qh, kvh, d], [graph, sinked], n) = (heads, flags, reqs.len());
    let (width, qw, scale) = (kvh * d, qh * d, (d as f32).sqrt().recip());
    let (mut qo, mut kv_indptr, mut kv_len) = (vec![0], vec![0], vec![]);
    let (mut last, mut pages, mut owner) = (vec![], vec![], vec![]);
    let mut next = 1usize;
    for (r, &(new, prefix)) in reqs.iter().enumerate() {
        let len = new + prefix;
        let np = len.div_ceil(PAGE);
        // Each request's pages run backwards through its stretch of the pool.
        pages.push((0..np).map(|i| next + np - 1 - i).collect::<Vec<_>>());
        next += np;
        qo.push(qo[r] + new as i32);
        kv_indptr.push(kv_indptr[r] + np as i32);
        kv_len.push(len as i32);
        last.push(((len - 1) % PAGE + 1) as i32);
        owner.extend((0..new).map(|t| (r, prefix + t + 1)));
    }
    let (rows, cells) = (qo[n] as usize, next * PAGE);
    let mut lcg = Lcg::seeded(0x5eed + d as u64);
    let (mut k_raw, mut k) = lcg.row(cells * width);
    let (mut v_raw, mut v) = lcg.row(cells * width);
    let (mut q_raw, mut q) = lcg.row(rows * qw);
    let [lk, lv] = live.map(|w| if w == 0 { d } else { w });
    for (raw, x, w) in [
        (&mut k_raw, &mut k, lk),
        (&mut v_raw, &mut v, lv),
        (&mut q_raw, &mut q, lk),
    ] {
        for i in (0..x.len()).filter(|i| i % d >= w) {
            (raw[i], x[i]) = (0, 0.0);
        }
    }
    let sinks: Vec<f32> = (0..qh).map(|h| 0.5 * (h % 7) as f32 - 1.0).collect();
    let sunk = sinked.then_some(&sinks[..]);
    let (want_o, want_lse) = attend([&q, &k, &v], [qh, kvh, d, d], scale, sunk, |row, _| {
        let (r, reach) = owner[row];
        let first = window.map_or(0, |w| reach.saturating_sub(w as usize));
        let cell = |t: usize| pages[r][t / PAGE] * PAGE + t % PAGE;
        (first..reach).map(cell).collect()
    });
    let mut gpu = Gpu::open();
    let flat: Vec<i32> = pages.iter().flatten().map(|&p| p as i32).collect();
    let ids = col(&mut gpu, &flat, I32);
    let indptr = col(&mut gpu, &kv_indptr, I32);
    let kvp = pool(ten(gpu.up(&k_raw), cells, width, Bf16), ids, indptr, PAGE);
    let values = ten(gpu.up(&v_raw), cells, width, Bf16);
    let last_page_lens = col(&mut gpu, &last, I32);
    let kvp = KvPool {
        values,
        last_page_lens,
        head_stride: d as i64,
        live: live.map(|w| w as u32),
        ..kvp
    };
    let data = ten(gpu.up(&q_raw), rows, qw, Bf16);
    let qt = RaggedTensor {
        data,
        indptr: col(&mut gpu, &qo, I32),
    };
    let (o_at, lse_at) = (gpu.zeros(rows * qw * 2), gpu.zeros(rows * qh * 4));
    // The tcgen05 prefill folds bf16 sinks in its epilogue.
    let sinks16: Vec<u16> = sinks.iter().map(|&v| crate::common::to_bf16(v)).collect();
    let sink_t = match tuning.prefill_tc5 {
        true => col(&mut gpu, &sinks16, Bf16),
        false => col(&mut gpu, &sinks, F32),
    };
    let stage = gpu.zeros(16);
    let (int_ptr, float_ptr, int_bytes) = (gpu.zeros(64 << 20), gpu.zeros(64 << 20), 64 << 20);
    let ws = Workspace {
        int_ptr,
        int_bytes,
        float_ptr,
        float_bytes: int_bytes,
    };
    let ctx = gpu.ctx().with_tuning(tuning);
    let device = Device::probe(&ctx).expect("the device answers");
    let (first, count) = lanes.flatten().unwrap_or((0, n as u32));
    if count != n as u32 {
        upload_to(stage, &[count, first, count, first]);
        ctx.arm_stage(stage);
    }
    let (a, b) = (first as usize, (first + count) as usize);
    let ([qh, kvh, d], page_size) = ([qh, kvh, d].map(|x| x as u32), PAGE as u32);
    let (num_q_heads, num_kv_heads, lane_offset) = (qh, kvh, first);
    #[rustfmt::skip]
    let shape = Shape { num_requests: count, lane_offset, num_q_heads, num_kv_heads, head_dim: d, page_size, hnd_layout: false };
    let (row_offset, live_rows) = (qo[a] as u32, (qo[b] - qo[a]) as u32);
    let live = Live {
        requests: count,
        lane_offset,
        row_offset,
        rows: live_rows,
    };
    let (o, lse) = (
        &mut ten(o_at, rows, qw, Bf16),
        &mut ten(lse_at, rows, qh as usize, F32),
    );
    let toggles = Toggles::of(tuning);
    let (ip, kl, total) = (&kv_indptr[a..=b], &kv_len[a..b], rows as u32);
    // Twice off one staged plan: one fire's workspace must not spoil the next.
    for _ in 0..2 {
        gpu.sync();
        upload_to(o_at, &vec![0x7f7fu16; rows * qw]);
        let fired = if lanes.is_some() {
            let grid = grid_of(d, qh, kvh, &device);
            let plan = plan::plan_decode(
                ip, kl, shape, live, window, graph, grid, toggles, &device, ws,
            );
            let plan = plan.expect("the decode plans");
            plan.stage(&ctx).expect("the plan uploads");
            match sinked {
                true => {
                    attn::decode_lse_sink(&ctx, qt, &plan, &kvp, window, d, scale, o, lse, sink_t)
                }
                false => attn::decode_lse(&ctx, qt, &plan, &kvp, window, d, scale, o, lse),
            }
        } else {
            let (q, k, l) = (&qo, &kv_indptr, &kv_len);
            let p = plan::plan_prefill(
                q, k, l, total, shape, live, window, true, graph, None, &device, toggles, ws,
            );
            let plan = p.expect("the prefill plans");
            plan.stage(&ctx).expect("the plan uploads");
            let (w, s) = (window, scale);
            match sinked {
                true => attn::prefill_lse_sink(&ctx, qt, &plan, &kvp, w, d, kvh, s, o, lse, sink_t),
                false => attn::prefill_lse(&ctx, qt, &plan, &kvp, w, d, kvh, s, o, lse),
            }
        };
        fired.expect("the attention fires");
    }
    let what = format!("{qh}/{kvh} heads of {d}, {window:?}, {graph} {sinked} {lanes:?}");
    // Rows outside the plan's lanes keep the scrub.
    let (ra, rb, mut want_o) = (
        row_offset as usize,
        (row_offset + live_rows) as usize,
        want_o,
    );
    want_o[..ra * qw].fill(from_bf16(0x7f7f));
    want_o[rb * qw..].fill(from_bf16(0x7f7f));
    // A padded head's values past their width are never written.
    want_o
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| i % d as usize >= lv)
        .for_each(|(_, o)| *o = from_bf16(0x7f7f));
    landed(&gpu, o_at, &want_o, 2e-2, 1.0, &what);
    let got_lse: Vec<f32> = gpu.down(lse_at, rows * qh as usize);
    let span = ra * qh as usize..rb * qh as usize;
    near(&got_lse[span.clone()], &want_lse[span], 1e-2, 1.0, &what);
}

#[test]
fn the_paged_attention_answers_the_reference() {
    let lens = [1usize, 15, 16, 17, 130, 300, 701];
    let dec: &[_] = &lens.map(|l| (1, l - 1));
    let chunks: &[_] = &[(70, 0), (33, 17), (200, 40), (129, 0)];
    let long: &[_] = &[(800, 40), (600, 0)];
    let (own, rec, sunk, both) = ([false; 2], [true, false], [false, true], [true; 2]);
    let (all, pre, lanes, more) = (Some(None), None, Some(Some((2, 3))), Some(Some((5, 2))));
    let (fa2, w128, w200, w100) = (Tuning::DEFAULT, Some(128), Some(200), Some(100));
    #[rustfmt::skip]
    let cases = [
        // Each decode read once, every head width, whole and windowed, own and recorded plans.
        ([64, 8, 64], dec, None, own, all, read(false, false, 0, 0)),
        ([32, 16, 256], dec, w200, both, all, read(true, false, 0, 0)),
        ([32, 4, 512], dec, None, rec, all, read(false, true, 0, 0)),
        ([8, 8, 128], dec, w200, sunk, all, read(true, true, 0, 0)),
        ([64, 8, 64], dec, w128, both, lanes, read(false, true, 256, 0)),
        ([64, 8, 64], dec, w200, own, all, read(true, true, 256, 0)),
        ([32, 16, 256], dec, None, sunk, all, read(false, true, 0, 8)),
        ([64, 8, 64], dec, w128, both, more, read(true, true, 256, 2)),
        ([32, 4, 512], dec, w200, rec, all, read(false, true, 0, 1)),
        // FA2's prefill, one row a request and chunks over cached prefixes, sinked and not.
        ([64, 8, 64], dec, w200, both, pre, read(true, true, 0, 0)),
        ([32, 4, 512], chunks, None, own, pre, fa2),
        ([8, 2, 128], chunks, w100, sunk, pre, fa2),
        // B200's tcgen05 prefill: whole, over a short window (its blocks each walking several tiles), and
        // a 64-wide head's tile pairs.
        ([32, 4, 128], chunks, None, rec, pre, tuned(|t| t.prefill_tc5 = true)),
        ([64, 8, 256], long, w100, sunk, pre, tuned(|t| (t.prefill_tc5, t.prefill_wide_sliding) = (true, true))),
        ([128, 16, 64], long, Some(300), sunk, pre, tuned(|t| t.prefill_tc5 = true)),
    ];
    for (heads, reqs, window, flags, lanes, tuning) in cases {
        check(heads, reqs, window, flags, lanes, tuning);
    }
    // MiMo's heads, 192-wide keys and 128-wide values padded to 256: the
    // tcgen05 prefill reads the widths alone, whole and over a window.
    let tc5 = tuned(|t| (t.prefill_tc5, t.prefill_wide_sliding) = (true, true));
    check_live([64, 8, 256], long, None, own, pre, tc5, [192, 128]);
    check_live([64, 8, 256], long, w128, sunk, pre, tc5, [192, 128]);
}
