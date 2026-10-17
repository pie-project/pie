#![cfg(feature = "cuda")]

//! `attn_ragged::ragged` against the host softmax: each mask arm, each head
//! width, cross tables, an armed seat and a captured graph's replays.

use crate::common::attn_ref::{ten, wide};
use crate::common::host::{assert_near, indptr, ragged_attention, upload_to};
use crate::common::{Gpu, Lcg, captured, forget, replay};
use dtype::Dtype::{Bf16, F32, I32, U8};
use kernels_cuda::attn_ragged::{RaggedMask, ragged};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mask {
    None,
    Bias,
    SelfOnly,
    Tags,
    Class,
}

const MAX_LEN: usize = 130;
const REF_START: [i32; 6] = [20, 100, 200, -1, 0, 3];

/// `extra` plane rows past the table hold noise the arm, under an armed
/// seat, must leave alone.
fn check(heads: [usize; 3], q_sizes: &[u32], kv_sizes: &[u32], mask: Mask, extra: usize) {
    let ([qh, kvh, hd], qt, kt) = (heads, indptr(q_sizes), indptr(kv_sizes));
    let (qr, kr) = (qt[qt.len() - 1] as usize + extra, kt[kt.len() - 1] as usize);
    let mut lcg = Lcg::seeded((hd + qr) as u64);
    let ((q_raw, q), (k_raw, k), (v_raw, v)) = (
        lcg.row(qr * qh * hd),
        lcg.row(kr * kvh * hd),
        lcg.row(kr * kvh * hd),
    );
    let (fill, span) = (lcg.row(qr * qh * hd).0, 2 * MAX_LEN - 1);
    let bias: Vec<f32> = (0..qh * span).map(|_| 2.0 * lcg.unit()).collect();
    let tags: Vec<i32> = (0..qr as i32)
        .map(|r| if r % 9 < 4 { -1 } else { r / 3 })
        .collect();
    let classes: Vec<i32> = (0..qr as i32).map(|r| r % 3).collect();
    let allowed: Vec<u8> = (0..9).map(|c| u8::from(c % 4 == 0 || c % 3 == 1)).collect();
    let scale = 2.0 / (hd as f32).sqrt();
    let want = ragged_attention([&q, &k, &v], &qt, &kt, heads, scale, |g, h, r, j| {
        let start = qt[g] as usize + REF_START[g].max(0) as usize;
        let keep = match mask {
            Mask::None | Mask::Bias => true,
            // A group's rows from its reference start read only those rows.
            Mask::SelfOnly => r < start || j >= start,
            Mask::Tags => tags[r] < 0 || tags[j] == tags[r],
            Mask::Class => allowed[(classes[r] * 3 + classes[j]) as usize] == 1,
        };
        let d = (j + MAX_LEN - 1).saturating_sub(r).min(span - 1);
        keep.then(|| {
            if mask == Mask::Bias {
                bias[h * span + d]
            } else {
                0.0
            }
        })
    });
    let mut gpu = Gpu::open();
    let o_at = gpu.up(&fill);
    // Every table ends where mapped memory ends: a padded query tile's read
    // past one faults, as it once read a neighbour's bytes.
    let mut edge = |v: &[i32]| ten(gpu.up_at_edge(v), v.len(), 1, I32);
    let (qtab, ktab, tag_t, class_t, refs) = (
        edge(&qt),
        edge(&kt),
        edge(&tags),
        edge(&classes),
        edge(&REF_START),
    );
    let arm = match mask {
        Mask::None => RaggedMask::None,
        Mask::Bias => RaggedMask::RelativeBias {
            table: ten(gpu.up(&bias), qh, span, F32),
            max_len: MAX_LEN as u32,
        },
        Mask::SelfOnly => RaggedMask::ReferenceSelfOnly { ref_start: refs },
        Mask::Tags => RaggedMask::ReferenceTags {
            q_tags: tag_t,
            kv_tags: tag_t,
        },
        Mask::Class => {
            let table = ten(gpu.up_at_edge(&allowed), 9, 1, U8);
            RaggedMask::ClassTable {
                q_classes: class_t,
                kv_classes: class_t,
                table,
                count: 3,
            }
        }
    };
    let (qp, kp, vp) = (
        ten(gpu.up(&q_raw), qr, qh * hd, Bf16),
        ten(gpu.up(&k_raw), kr, kvh * hd, Bf16),
        ten(gpu.up(&v_raw), kr, kvh * hd, Bf16),
    );
    let (ctx, mut o, hd32) = (gpu.ctx(), ten(o_at, qr, qh * hd, Bf16), hd as u32);
    if extra > 0 {
        ctx.arm_stage(gpu.up(&[qr as u32, 0, 2, 1]));
    }
    ragged(&ctx, qp, kp, vp, qtab, ktab, hd32, scale, arm, &mut o).expect("the arm fires");
    gpu.sync();
    let (got, past) = (gpu.down::<u16>(o_at, qr * qh * hd), (qr - extra) * qh * hd);
    assert_near(
        &got[..past],
        &want[..past],
        1e-2,
        &format!("{heads:?} {mask:?}"),
    );
    assert_eq!(
        got[past..],
        fill[past..],
        "{mask:?}: a row past every group moved"
    );
}

#[test]
fn the_ragged_arm_answers_the_reference() {
    let (q_cross, kv_cross) = (&[7, 0, 200, 300, 3], &[300, 40, 0, 500, 1]);
    let (lanes, refs, seat) = (
        &[7u32, 11, 21],
        &[64u32, 300, 200, 50, 40, 5],
        &[40, 100, 300, 64],
    );
    // Cross tables with empty groups at 64 and 128, the bias at full heads,
    // each lane mask, and every group served under an armed seat at 256.
    check([4, 2, 64], q_cross, kv_cross, Mask::None, 0);
    check([4, 2, 128], q_cross, kv_cross, Mask::None, 0);
    check([64, 16, 64], &[37, 130, 64], &[37, 130, 64], Mask::Bias, 0);
    check([4, 2, 128], refs, refs, Mask::SelfOnly, 0);
    check([2, 1, 256], lanes, lanes, Mask::Tags, 0);
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
    let (q_at, k_at, v_at) = (
        gpu.zeros(rows * qw * 2),
        gpu.zeros(rows * kw * 2),
        gpu.zeros(rows * kw * 2),
    );
    let (t_at, o_at, e_at) = (
        gpu.up(&table),
        gpu.zeros(rows * qw * 2),
        gpu.zeros(rows * qw * 2),
    );
    let (win_at, ctx, groups) = (
        gpu.up(&[rows as u32, 0, 3, 0]),
        gpu.ctx(),
        ten(t_at, 4, 1, I32),
    );
    let (q, k, v, none) = (
        ten(q_at, rows, qw, Bf16),
        ten(k_at, rows, kw, Bf16),
        ten(v_at, rows, kw, Bf16),
        RaggedMask::None,
    );
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
    fired
        .expect("the capture ran")
        .expect("the arm fires under capture");
    ctx.disarm_stage();
    let mut before: Vec<u16> = Vec::new();
    for (pass, window) in [[rows as u32, 0, 3, 0], [rows as u32, 0, 2, 1]]
        .iter()
        .enumerate()
    {
        fresh(&gpu);
        let mut groups = table.clone();
        groups[0] = groups[pass];
        (upload_to(win_at, window), upload_to(t_at, &groups));
        fire(e_at).expect("the eager twin");
        replay(&ctx, exec);
        gpu.sync();
        let (want, got) = (
            wide(&gpu.down::<u16>(e_at, rows * qw)),
            gpu.down::<u16>(o_at, rows * qw),
        );
        let split = groups[0] as usize * qw;
        assert_near(
            &got[split..],
            &want[split..],
            1e-2,
            &format!("replay {pass}"),
        );
        assert_eq!(
            got[..split],
            before[..split.min(before.len())],
            "group 0 sits outside the window"
        );
        before = got;
    }
    forget(exec);
}
