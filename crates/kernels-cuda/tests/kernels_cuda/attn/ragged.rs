#![cfg(feature = "cuda")]

//! `ragged::ragged` against the host softmax: each mask arm, each head
//! width, cross tables, an armed seat and a captured graph's replays.

use crate::common::attn_ref::{landed, ten, wide};
use crate::common::host::{indptr, ragged_attention, upload_to};
use crate::common::{Gpu, Lcg, captured, forget, replay};
use dtype::Dtype::{Bf16, F32, I32, U8};
use kernels_cuda::attn::ragged::{RaggedMask, ragged};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mask {
    None,
    Bias,
    Class,
}

const MAX_LEN: usize = 130;

/// `extra` plane rows past the table hold noise the arm, under an armed
/// seat, must leave alone.
fn check(heads: [usize; 3], q_sizes: &[u32], kv_sizes: &[u32], mask: Mask, extra: usize) {
    let ([qh, kvh, hd], qt, kt) = (heads, indptr(q_sizes), indptr(kv_sizes));
    let (qr, kr) = (qt[qt.len() - 1] as usize + extra, kt[kt.len() - 1] as usize);
    let mut lcg = Lcg::seeded((hd + qr) as u64);
    let (q_raw, q) = lcg.row(qr * qh * hd);
    let (k_raw, k) = lcg.row(kr * kvh * hd);
    let (v_raw, v) = lcg.row(kr * kvh * hd);
    let (fill, span) = (lcg.row(qr * qh * hd).0, 2 * MAX_LEN - 1);
    let bias: Vec<f32> = (0..qh * span).map(|_| 2.0 * lcg.unit()).collect();
    let classes: Vec<i32> = (0..qr as i32).map(|r| r % 3).collect();
    let allowed: Vec<u8> = (0..9).map(|c| u8::from(c % 4 == 0 || c % 3 == 1)).collect();
    let scale = 2.0 / (hd as f32).sqrt();
    let mut want = ragged_attention([&q, &k, &v], &qt, &kt, heads, scale, |_, h, r, j| {
        let keep = match mask {
            Mask::None | Mask::Bias => true,
            Mask::Class => allowed[(classes[r] * 3 + classes[j]) as usize] == 1,
        };
        let d = (j + MAX_LEN - 1).saturating_sub(r).min(span - 1);
        let bias = bias[h * span + d] * f32::from(u8::from(mask == Mask::Bias));
        keep.then_some(bias)
    });
    let mut gpu = Gpu::open();
    let o_at = gpu.up(&fill);
    // Every table ends where mapped memory ends: a padded query tile's read
    // past one faults, as it once read a neighbour's bytes.
    let mut edge = |v: &[i32]| ten(gpu.up_at_edge(v), v.len(), 1, I32);
    let (qtab, ktab, class_t) = (edge(&qt), edge(&kt), edge(&classes));
    let (table, max_len) = (ten(gpu.up(&bias), qh, span, F32), MAX_LEN as u32);
    let allowed = ten(gpu.up_at_edge(&allowed), 9, 1, U8);
    let arm = match mask {
        Mask::None => RaggedMask::None,
        Mask::Bias => RaggedMask::RelativeBias { table, max_len },
        #[rustfmt::skip]
        Mask::Class => RaggedMask::ClassTable { q_classes: class_t, kv_classes: class_t, table: allowed, count: 3 },
    };
    let qp = ten(gpu.up(&q_raw), qr, qh * hd, Bf16);
    let kp = ten(gpu.up(&k_raw), kr, kvh * hd, Bf16);
    let vp = ten(gpu.up(&v_raw), kr, kvh * hd, Bf16);
    let (ctx, o, hd32) = (gpu.ctx(), &mut ten(o_at, qr, qh * hd, Bf16), hd as u32);
    if extra > 0 {
        ctx.arm_stage(gpu.up(&[qr as u32, 0, 2, 1]));
    }
    ragged(&ctx, qp, kp, vp, qtab, ktab, hd32, scale, arm, o).expect("the arm fires");
    // A row past every group keeps its noise.
    let past = (qr - extra) * qh * hd;
    want[past..].copy_from_slice(&wide(&fill[past..]));
    landed(&gpu, o_at, &want, 1e-2, 1.0, &format!("{heads:?} {mask:?}"));
}

#[test]
fn the_ragged_arm_answers_the_reference() {
    let (q_cross, kv_cross) = (&[7, 0, 200, 300, 3], &[300, 40, 0, 500, 1]);
    let (lanes, seat) = (&[7u32, 11, 21], &[40, 100, 300, 64]);
    let bias = &[37u32, 130, 64];
    // Cross tables with empty groups at 64 and 128, the bias at full heads,
    // each lane mask, and every group served under an armed seat at 256.
    check([4, 2, 64], q_cross, kv_cross, Mask::None, 0);
    check([4, 2, 128], q_cross, kv_cross, Mask::None, 0);
    check([64, 16, 64], bias, bias, Mask::Bias, 0);
    check([2, 1, 256], lanes, lanes, Mask::Class, 0);
    check([4, 2, 64], lanes, lanes, Mask::Class, 0);
    check([2, 2, 256], seat, seat, Mask::None, 96);
}

/// A captured arm replays over new values, and over a seat whose window
/// leaves its first group out (that group keeps the last replay's bytes).
#[test]
fn the_ragged_arm_replays_from_a_captured_graph() {
    let (table, qw, kw) = (indptr(&[200, 130, 300]), 4 * 128, 2 * 128);
    let rows = table[3] as usize;
    let mut gpu = Gpu::open();
    let (q_at, o_at, e_at) = (
        gpu.zeros(rows * qw * 2),
        gpu.zeros(rows * qw * 2),
        gpu.zeros(rows * qw * 2),
    );
    let (k_at, v_at, t_at) = (
        gpu.zeros(rows * kw * 2),
        gpu.zeros(rows * kw * 2),
        gpu.up(&table),
    );
    let (win_at, ctx, groups) = (
        gpu.up(&[rows as u32, 0, 3, 0]),
        gpu.ctx(),
        ten(t_at, 4, 1, I32),
    );
    let (q, k, v) = (
        ten(q_at, rows, qw, Bf16),
        ten(k_at, rows, kw, Bf16),
        ten(v_at, rows, kw, Bf16),
    );
    let none = RaggedMask::None;
    let fire = |o: u64| {
        ragged(
            &ctx,
            q,
            k,
            v,
            groups,
            groups,
            128,
            0.0,
            none,
            &mut ten(o, rows, qw, Bf16),
        )
    };
    let mut lcg = Lcg::seeded(0x9a);
    let mut fresh = |gpu: &Gpu| {
        gpu.sync();
        for (at, n) in [(q_at, qw), (k_at, kw), (v_at, kw)] {
            upload_to(at, &lcg.row(rows * n).0);
        }
    };
    fresh(&gpu);
    ctx.arm_stage(win_at);
    fire(o_at).expect("the warm fire");
    gpu.sync();
    let mut fired = None;
    let exec = captured(&ctx, || fired = Some(fire(o_at)));
    fired.expect("the capture ran").expect("the arm fires");
    ctx.disarm_stage();
    let mut before = vec![0u16; rows * qw];
    for (pass, window) in [[rows as u32, 0, 3, 0], [rows as u32, 0, 2, 1]]
        .iter()
        .enumerate()
    {
        fresh(&gpu);
        let mut groups = table.clone();
        groups[0] = groups[pass];
        upload_to(win_at, window);
        upload_to(t_at, &groups);
        fire(e_at).expect("the eager twin");
        replay(&ctx, exec);
        gpu.sync();
        let split = groups[0] as usize * qw;
        let mut want = wide(&gpu.down::<u16>(e_at, rows * qw));
        want[..split].copy_from_slice(&wide(&before[..split]));
        landed(&gpu, o_at, &want, 1e-2, 1.0, &format!("replay {pass}"));
        before = gpu.down(o_at, rows * qw);
    }
    forget(exec);
}
