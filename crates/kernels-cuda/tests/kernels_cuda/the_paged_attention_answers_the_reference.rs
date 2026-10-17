#![cfg(feature = "cuda")]

//! The paged attention launchers (`decode_lse`, `prefill_lse`, their sinked
//! forms and the tcgen05 `prefill`) against the host softmax, one case a
//! route.

use crate::common::attn_ref::{attend, col, near, pool, ten, tuned, wide};
use crate::common::{Gpu, Lcg, from_bf16, host::upload_to};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::attn::{self, fa2::decode_max_grid_size as grid_of};
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// The decode, over lanes `(first, count)` of a recorded plan where given.
    Decode(Option<(u32, u32)>),
    /// FA2's prefill with its log-sum-exp.
    Lse,
    /// The tcgen05 prefill.
    Tc5,
}

/// A decode read: the warp merge, the tensor-core decode, its whole window
/// and its cluster.
fn read(warp: bool, xqa: bool, whole: u32, cluster: u32) -> Tuning {
    tuned(|t| {
        (t.merge_warp, t.decode_xqa) = (warp, xqa);
        (t.decode_xqa_whole_window, t.decode_xqa_cluster) = (whole, cluster);
    })
}

/// `[q heads, kv heads, head dim]` over requests of `(new, cached)` rows.
#[allow(clippy::too_many_lines)]
fn check(
    [qh, kvh, d]: [usize; 3],
    reqs: &[(usize, usize)],
    window: Option<u32>,
    [graph, sinked]: [bool; 2],
    mode: Mode,
    tuning: Tuning,
) {
    let n = reqs.len();
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
    let (k_raw, k) = lcg.row(cells * width);
    let (v_raw, v) = lcg.row(cells * width);
    let (q_raw, q) = lcg.row(rows * qw);
    let sinks: Vec<f32> = (0..qh).map(|h| 0.5 * (h % 7) as f32 - 1.0).collect();
    let sunk = sinked.then_some(&sinks[..]);
    let (want_o, want_lse) = attend([&q, &k, &v], [qh, kvh, d, d], scale, sunk, |row, _| {
        let (r, reach) = owner[row];
        let first = window.map_or(0, |w| reach.saturating_sub(w as usize));
        (first..reach)
            .map(|t| pages[r][t / PAGE] * PAGE + t % PAGE)
            .collect()
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
        ..kvp
    };
    let data = ten(gpu.up(&q_raw), rows, qw, Bf16);
    let qt = RaggedTensor {
        data,
        indptr: col(&mut gpu, &qo, I32),
    };
    let (o_at, lse_at) = (gpu.zeros(rows * qw * 2), gpu.zeros(rows * qh * 4));
    let (stage, sink_t) = (gpu.zeros(16), col(&mut gpu, &sinks, F32));
    let (int_ptr, float_ptr, int_bytes) = (gpu.zeros(64 << 20), gpu.zeros(64 << 20), 64 << 20);
    let ws = Workspace {
        int_ptr,
        int_bytes,
        float_ptr,
        float_bytes: int_bytes,
    };
    let ctx = gpu.ctx().with_tuning(tuning);
    let device = Device::probe(&ctx).expect("the device answers");
    let (first, count) = if let Mode::Decode(Some(l)) = mode {
        l
    } else {
        (0, n as u32)
    };
    if count != n as u32 {
        upload_to(stage, &[count, first, count, first]);
        ctx.arm_stage(stage);
    }
    let (a, b) = (first as usize, (first + count) as usize);
    let ([qh, kvh, d], page_size) = ([qh, kvh, d].map(|x| x as u32), PAGE as u32);
    let (num_q_heads, num_kv_heads, lane_offset) = (qh, kvh, first);
    let shape = Shape {
        num_requests: count,
        lane_offset,
        num_q_heads,
        num_kv_heads,
        head_dim: d,
        page_size,
        hnd_layout: false,
    };
    let (row_offset, live_rows) = (qo[a] as u32, (qo[b] - qo[a]) as u32);
    let live = Live {
        requests: count,
        lane_offset,
        row_offset,
        rows: live_rows,
    };
    let mut o = ten(o_at, rows, qw, Bf16);
    let mut lse = ten(lse_at, rows, qh as usize, F32);
    let plain = Tuning {
        prefill_tc5: false,
        prefill_wide_sliding: false,
        ..tuning
    };
    let toggles = Toggles::of(plain);
    let (ip, kl, total) = (&kv_indptr[a..=b], &kv_len[a..b], rows as u32);
    // Twice off one staged plan: one fire's workspace must not spoil the next.
    for _ in 0..2 {
        gpu.sync();
        upload_to(o_at, &vec![0x7f7fu16; rows * qw]);
        let fired = if let Mode::Decode(_) = mode {
            let grid = grid_of(d, qh, kvh, &device);
            let plan = plan::plan_decode(
                ip, kl, shape, live, window, graph, grid, toggles, &device, ws,
            );
            let plan = plan.expect("the decode plans");
            plan.stage(&ctx).expect("the plan uploads");
            let (o, lse) = (&mut o, &mut lse);
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
            let (o, lse) = (&mut o, &mut lse);
            match (mode, sinked) {
                (Mode::Tc5, _) => attn::prefill(&ctx, qt, &plan, &kvp, window, d, kvh, scale, o),
                (_, true) => attn::prefill_lse_sink(
                    &ctx, qt, &plan, &kvp, window, d, kvh, scale, o, lse, sink_t,
                ),
                _ => attn::prefill_lse(&ctx, qt, &plan, &kvp, window, d, kvh, scale, o, lse),
            }
        };
        fired.expect("the attention fires");
    }
    gpu.sync();
    let got = wide(&gpu.down::<u16>(o_at, rows * qw));
    let got_lse: Vec<f32> = gpu.down(lse_at, rows * qh as usize);
    let (ra, rb) = (row_offset as usize, (row_offset + live_rows) as usize);
    let what = format!("{qh}/{kvh} heads of {d}, {window:?}, {graph} {sinked} {mode:?}");
    let (o_rows, l_rows) = (ra * qw..rb * qw, ra * qh as usize..rb * qh as usize);
    near(&got[o_rows.clone()], &want_o[o_rows], 2e-2, 1.0, &what);
    if !matches!(mode, Mode::Tc5) {
        near(
            &got_lse[l_rows.clone()],
            &want_lse[l_rows],
            1e-2,
            1.0,
            &what,
        );
    }
    let mut kept = got[..ra * qw].iter().chain(&got[rb * qw..]);
    let scrubbed = kept.all(|&x| x == from_bf16(0x7f7f));
    assert!(scrubbed, "{what}: a row outside the lanes moved");
}

#[test]
fn the_paged_attention_answers_the_reference() {
    let lens = [1usize, 15, 16, 17, 130, 300, 701];
    let dec: &[_] = &lens.map(|l| (1, l - 1));
    let chunks: &[_] = &[(70, 0), (33, 17), (200, 40), (129, 0)];
    let two: &[_] = &[(300, 0), (90, 120)];
    let tall: &[_] = &[(1, 300), (256, 1793), (1, 47), (3, 20)];
    let few: &[_] = &[(150, 0), (90, 12)];
    let (own, rec, sunk, both) = ([false; 2], [true, false], [false, true], [true; 2]);
    let (d0, lse, tc5) = (Mode::Decode(None), Mode::Lse, Mode::Tc5);
    let (lanes, more) = (Mode::Decode(Some((2, 3))), Mode::Decode(Some((5, 2))));
    let (fa2, w128, w200, w100) = (Tuning::DEFAULT, Some(128), Some(200), Some(100));
    let tc5_on = |w| tuned(|t| (t.prefill_tc5, t.prefill_wide_sliding) = (true, w));
    #[rustfmt::skip]
    let cases = [
        // Each decode read once, every head width, whole and windowed, own and recorded plans.
        ([64, 8, 64], dec, None, own, d0, read(false, false, 0, 0)),
        ([32, 16, 256], dec, w200, both, d0, read(true, false, 0, 0)),
        ([32, 4, 512], dec, None, rec, d0, read(false, true, 0, 0)),
        ([8, 8, 128], dec, w200, sunk, d0, read(true, true, 0, 0)),
        ([64, 8, 64], dec, w128, both, lanes, read(false, true, 256, 0)),
        ([64, 8, 64], dec, w200, own, d0, read(true, true, 256, 0)),
        ([32, 16, 256], dec, None, sunk, d0, read(false, true, 0, 8)),
        ([64, 8, 64], dec, w128, both, more, read(true, true, 256, 2)),
        ([32, 4, 512], dec, w200, rec, d0, read(false, true, 0, 1)),
        // FA2's prefill, one row a request and chunks over cached prefixes, sinked and not.
        ([64, 8, 64], dec, w200, both, lse, read(true, true, 0, 0)),
        ([32, 4, 512], chunks, None, own, lse, fa2),
        ([8, 2, 128], chunks, w100, sunk, lse, fa2),
        // The tcgen05 prefill at each head width, windowed on both sliding forms.
        ([8, 2, 128], chunks, None, own, tc5, tc5_on(false)),
        ([8, 4, 256], two, w100, own, tc5, tc5_on(false)),
        ([8, 4, 256], two, w100, own, tc5, tc5_on(true)),
        ([3, 3, 64], few, None, own, tc5, tc5_on(false)),
        ([32, 4, 512], tall, None, own, tc5, tc5_on(false)),
    ];
    for (heads, reqs, window, flags, mode, tuning) in cases {
        check(heads, reqs, window, flags, mode, tuning);
    }
}
