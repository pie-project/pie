//! The tcgen05 GEMM, dense and grouped (contiguous and masked), on each
//! operand kind, plan and epilogue, against the host's dot of its decoded
//! operands; split, dependent and captured launches land their bits again.
//! (Its quantized landings are the routed MLP's, read there.)

use crate::common::linear_ref::{Bench, TOL, affine, check, dot, glu_want, pick, tab};
use crate::common::tc5::Side;
use crate::common::{captured, forget, replay};
use kernels_cuda::Ctx;
use kernels_cuda::linear::tc5::{
    Elem, Epilogue, Glu, Grouped, Groups, Operand, Plan, gemm_epi_with, gemm_with,
    grouped_gemm_epi_with, supported,
};
use kernels_cuda::tuning::Tuning;

type Spec = (Elem, Option<usize>);
const BF: Spec = (Elem::Bf16, None);
const F8: Spec = (Elem::E4m3, None);
const DS: Spec = (Elem::E4m3, Some(128));
const MX: Spec = (Elem::E4m3, Some(32));
const NV: Spec = (Elem::E2m1, Some(16));
const W4: Spec = (Elem::E2m1, Some(32));
const NAN: u16 = 0x7fc0;

/// A stated plan: swapped, paired, its Q tile and its K splits.
#[rustfmt::skip]
fn p(swap: bool, pair: bool, bn: u32, splits: u32) -> Option<Plan> {
    Some(Plan { swap, pair, bn, stages: 0, splits })
}

/// An operand's device form: a W4A16 weight's scales row-major (NVFP4's
/// tensor scale a row folded into its values), any other's in atoms, row
/// `r` at atom row `at(r)`.
fn operand(
    b: &mut Bench,
    s: &mut Side,
    (elem, gran): Spec,
    mixed: bool,
    k: usize,
    at: impl Fn(usize) -> usize,
) -> (Operand, u64) {
    let (ptr, rows) = (b.up(&s.bytes), s.values.len() / k);
    let Some(gran) = gran else {
        return (Operand::plain(ptr, elem), 0);
    };
    let mut scaled = |v: &[u8]| Operand::scaled(ptr, elem, b.up(v), gran as u32);
    if !mixed {
        return (scaled(&s.atoms_at(k, at(rows - 1) + 1, at)), 0);
    }
    let op = scaled(&s.codes);
    let ts = tab(rows, |r| pick(gran == 16, 0.5 + (r % 8) as f32 * 0.25, 1.0));
    for (i, v) in s.values.iter_mut().enumerate() {
        *v *= ts[i / k];
    }
    (op, pick(gran == 16, b.up(&ts), 0))
}

/// A dense product of `m` rows, or `e` experts' contiguous, or masked
/// (`m_max` rows each).
#[derive(Clone, Copy, Debug)]
enum Lay {
    D(usize),
    C(usize),
    M(usize, usize),
}

/// One case: the operands (an affine weight `(U4, group)`), the layout,
/// `[n, k]`, the GLU, a row scale, the plan (`None`: chosen) and the tuning
/// (bit 0: NVFP4 at 128 K a stage; 1: split tiles summed by their last
/// split; 2: the launch repeated and replayed).
#[rustfmt::skip]
type Case = (Spec, Spec, Lay, [usize; 2], Option<Glu>, bool, Option<Plan>, u8);

fn run((a, w, lay, [n, k], f, rs, plan, tun): Case) {
    let what = format!("{a:?} x {w:?} {lay:?} {n}x{k} {f:?} {rs} {plan:?} {tun}");
    let mut b = Bench::new(0xe9 ^ (n * k) as u64);
    let (e, mm, dense) = match lay {
        Lay::D(m) => (1, m, true),
        Lay::C(e) => (e, 0, false),
        Lay::M(e, mm) => (e, mm, false),
    };
    // Each expert's rows: none, a few, a block and a half, ...; at offsets
    // in whole 64s contiguous, in `m_max`s masked.
    let count = |x: usize| {
        pick(
            dense,
            mm,
            [0, 5, 150, 37, 64, 1][x % 6].min(pick(mm > 0, mm, 300)),
        )
    };
    let mut off = vec![0usize];
    for x in 0..e {
        off.push(off[x] + pick(mm > 0, mm, count(x).next_multiple_of(64)));
    }
    let rows = off[e] + pick(mm > 0, 0, 128);
    let (mut rows_of, mut past) = (Vec::new(), vec![(off[e], rows)]);
    for x in 0..e {
        rows_of.extend((0..count(x)).map(|i| (off[x] + i, x)));
        past.extend((mm > 0).then_some((off[x] + count(x), off[x + 1] + usize::from(dense))));
    }
    let (offs, counts) = (tab(e + 1, |x| off[x] as i32), tab(e, |x| count(x) as i32));
    #[rustfmt::skip]
    let groups = match lay {
        Lay::D(_) => None,
        Lay::C(_) => Some(Groups::Contiguous { offsets: b.up(&offs), rows: rows as u32 }),
        Lay::M(..) => Some(Groups::Masked { counts: b.up(&counts), m_max: mm as u32 }),
    };
    let (wrows, mixed) = (
        n << u8::from(f.is_some()),
        a.0 == Elem::Bf16 && w.1.is_some(),
    );
    let (w_op, wt, ts) = if w.0 == Elem::U4 {
        let g = w.1.expect("a group");
        let (s, fs, zs) = affine(&mut b.lcg, [e * wrows, k, g, 4]);
        let op = Operand::affine(b.up(&s.bytes), b.up(&fs), b.up(&zs), g as u32);
        (op, s, 0)
    } else {
        let mut s = b.side(w.0, w.1, e * wrows, k);
        let (op, ts) = operand(&mut b, &mut s, w, mixed, k, |r| r);
        (op, s, ts)
    };
    // A masked activation's scale atoms pad each expert to whole blocks.
    let at = |r: usize| {
        pick(
            dense || mm == 0,
            r,
            r / mm.max(1) * mm.next_multiple_of(128) + r % mm.max(1),
        )
    };
    let mut act = b.side(a.0, a.1, rows, k);
    let (a_op, _) = operand(&mut b, &mut act, a, false, k, at);
    let (alpha, ae) = (
        pick(mixed, 1.0, 0.5),
        tab(e, |x| 0.5 + 0.25 * (x % 3) as f32),
    );
    let ((bias, bv), rsv) = (
        b.bf16(e * wrows),
        tab(rows, |r| 0.25 + (r % 7) as f32 / 8.0),
    );
    // Each routed row's dot, through the bias, the GLU and the row scale.
    let gain = |x: usize| f64::from(alpha * pick(dense || mixed, 1.0, ae[x]));
    let a_of = |j: usize, i: usize| f64::from(act.value(rows_of[j].0, k, i)) * gain(rows_of[j].1);
    let w_of = |j: usize, c: usize, i: usize| f64::from(wt.value(rows_of[j].1 * wrows + c, k, i));
    let d = dot([rows_of.len(), wrows, k], a_of, w_of);
    let bias_of = |j: usize, c: usize| pick(mixed, bv[rows_of[j].1 * wrows + c], 0.0);
    let v = |j: usize, c: usize| (d[j * wrows + c].0 + bias_of(j, c), d[j * wrows + c].1);
    let glu = |j: usize, c: usize| f.map_or_else(|| v(j, c), |f| glu_want(f, v(j, c), v(j, n + c)));
    let wants = tab(rows_of.len() * n, |i| {
        let ((x, s), r) = (
            glu(i / n, i % n),
            pick(rs, f64::from(rsv[rows_of[i / n].0]), 1.0),
        );
        (x * r, s * r)
    });
    let (y, row_scale, bias) = (
        b.up(&vec![NAN; (rows + 1) * n]),
        pick(rs, b.up(&rsv), 0),
        pick(mixed, bias, 0),
    );
    #[rustfmt::skip]
    let epi = Epilogue { glu: f, row_scale, bias, weight_scale: ts, ..Epilogue::STORE };
    let (k128, tickets) = (tun & 1 != 0, pick(tun & 2 != 0, 65_536, 0));
    #[rustfmt::skip]
    let ctx = b.ctx().with_tuning(Tuning { gemm_nvfp4_k128: k128, gemm_tc5_split_tickets: tickets, ..Tuning::DEFAULT });
    let (m, n32, k32, alpha_e) = (rows_of.len() as u32, n as u32, k as u32, b.up(&ae));
    let fire = |ctx: &Ctx, s: f32| {
        match groups {
            None => {
                let plan = plan.unwrap_or_else(|| Plan::choose(m, wrows as u32, k32, 148));
                gemm_epi_with(ctx, a_op, w_op, y, [m, n32, k32], alpha * s, epi, plan)
            }
            Some(groups) => {
                let alpha_e = pick(mixed, 0, alpha_e);
                let g = Grouped {
                    experts: e as u32,
                    groups,
                    alpha_e,
                };
                let plan = plan.unwrap_or_else(|| Plan::grouped(&g, n32, k32, m, 148));
                grouped_gemm_epi_with(ctx, a_op, w_op, y, g, [n32, k32], alpha * s, epi, plan)
            }
        }
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    };
    fire(&ctx, 1.0);
    let got = b.read(y, (rows + 1) * n);
    check(
        &tab(wants.len(), |i| got[rows_of[i / n].0 * n + i % n]),
        &wants,
        TOL,
        &what,
    );
    // Rows past the product's are not touched.
    for &(from, to) in past.iter().filter(|_| !mixed && f.is_none()) {
        let nan = got[from * n..to * n].iter().all(|v| v.is_nan());
        assert!(nan, "{what}: row {from} moved");
    }
    if tun & 4 != 0 {
        // The finalize of a K split a programmatic dependent, or each tile
        // summed by its last split: a finalize that read the slabs early
        // would add stale halves.
        let base = ctx.tuning();
        let pdl = b.ctx().with_tuning(Tuning { pdl: true, ..base });
        #[rustfmt::skip]
        let tickets = b.ctx().with_tuning(Tuning { gemm_tc5_split_tickets: 65_536, ..base });
        again(&b, &[pdl, tickets], (y, got.len()), &what, &fire);
    }
}

/// `fire`'s product at half the scale then whole, eagerly and from a
/// captured graph on each context, lands the `(y, len)` it holds again.
fn again(b: &Bench, ctxs: &[Ctx], (y, len): (u64, usize), what: &str, fire: &impl Fn(&Ctx, f32)) {
    let (first, bits) = (b.read(y, len), |v: Vec<f64>| tab(len, |i| v[i].to_bits()));
    let same = || bits(b.read(y, len)) == bits(first.clone());
    for ctx in ctxs {
        let twice = || [0.5, 1.0].into_iter().for_each(|s| fire(ctx, s));
        twice();
        assert!(same(), "{what}: again");
        let exec = captured(ctx, twice);
        for _ in 0..2 {
            replay(ctx, exec);
            assert!(same(), "{what}: replayed");
        }
        forget(exec);
    }
}

#[test]
fn the_tcgen05_gemm_answers_the_dequantized_dot_every_case() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let (sw, geglu, no) = (Some(Glu::Swiglu), Some(Glu::GegluTanh), None);
    let clamp = Some(Glu::SwigluClamp { limit: 7.0 });
    let alpha = Some(Glu::SwigluClampAlpha {
        limit: 7.0,
        alpha: 1.702,
    });
    let (situ, bare) = (
        Glu::Situ {
            beta: 4.0,
            cap: 25.0,
        },
        Glu::Situ {
            beta: 1.0,
            cap: 0.0,
        },
    );
    let (situ, bare) = (Some(situ), Some(bare));
    let (u32_, u64_, u128_) = (
        (Elem::U4, Some(32)),
        (Elem::U4, Some(64)),
        (Elem::U4, Some(128)),
    );
    let (f, t) = (false, true);
    #[rustfmt::skip]
    let cases: &[Case] = &[
        // Dense: each kind, each Q tile, swapped at decode widths, paired;
        // more tiles than SMs, lone and paired (the 256-wide tile's two
        // accumulator stages sharing columns).
        (BF, BF, Lay::D(300), [200, 520], no, f, p(f, f, 128, 0), 0),
        (BF, BF, Lay::D(1), [384, 512], no, f, None, 0),
        (BF, BF, Lay::D(257), [96, 136], no, f, p(f, t, 64, 0), 0),
        (BF, BF, Lay::D(1280), [4096, 256], no, f, p(f, f, 128, 0), 0),
        (F8, F8, Lay::D(200), [160, 400], no, f, p(f, f, 32, 0), 0),
        (DS, DS, Lay::D(129), [192, 640], no, f, p(f, f, 64, 0), 0),
        (DS, DS, Lay::D(300), [256, 1152], no, f, p(f, t, 256, 0), 0),
        (DS, DS, Lay::D(1280), [4096, 256], no, f, p(f, f, 256, 0), 0),
        (DS, DS, Lay::D(1280), [4096, 256], no, f, p(f, t, 256, 0), 0),
        (MX, MX, Lay::D(130), [128, 320], no, f, p(f, f, 128, 0), 0),
        (NV, NV, Lay::D(130), [512, 512], no, f, p(f, f, 256, 0), 0),
        (NV, NV, Lay::D(5), [384, 768], no, f, None, 0),
        (W4, W4, Lay::D(128), [128, 128], no, f, p(f, f, 128, 0), 0),
        (DS, W4, Lay::D(150), [192, 640], no, f, p(f, t, 64, 0), 0),
        (MX, W4, Lay::D(7), [256, 1024], no, f, None, 0),
        // Decode widths split along K (one by hand), repeated and replayed.
        (F8, F8, Lay::D(3), [256, 1024], no, f, None, 4),
        (DS, DS, Lay::D(64), [512, 8192], no, f, None, 4),
        (BF, BF, Lay::D(16), [512, 4096], no, f, None, 4),
        (DS, DS, Lay::D(128), [256, 4096], no, f, p(f, f, 128, 4), 4),
        // Dense epilogues: each GLU (GeGLU's 192 ends mid-block), a row scale.
        (BF, BF, Lay::D(130), [128, 256], clamp, f, p(f, f, 256, 0), 0),
        (BF, BF, Lay::D(33), [256, 256], alpha, f, p(t, f, 64, 0), 0),
        (BF, BF, Lay::D(300), [192, 512], geglu, f, p(f, f, 256, 0), 0),
        (BF, BF, Lay::D(20), [256, 512], bare, f, p(t, f, 32, 0), 0),
        (DS, W4, Lay::D(140), [256, 512], situ, f, p(f, f, 256, 0), 0),
        (DS, DS, Lay::D(129), [384, 640], alpha, t, p(f, f, 256, 0), 0),
        (NV, NV, Lay::D(9), [256, 512], clamp, t, p(t, f, 32, 0), 0),
        // W4A16: MXFP4, NVFP4 (a tensor scale a row) and affine weights in
        // each group, a bias a weight row.
        (BF, W4, Lay::D(200), [256, 576], no, f, p(f, f, 128, 0), 0),
        (BF, NV, Lay::D(24), [384, 2880], no, f, p(t, f, 32, 0), 0),
        (BF, u64_, Lay::D(130), [256, 512], sw, f, p(f, f, 128, 0), 0),
        (BF, u32_, Lay::D(200), [256, 512], no, f, p(f, f, 128, 0), 0),
        (BF, u128_, Lay::D(200), [256, 512], no, f, p(f, f, 128, 0), 0),
        // Grouped: NVFP4 at 128 K a stage, K splits (summed by the last
        // split), swapped and paired; a GLU and a row scale.
        (BF, BF, Lay::C(8), [256, 512], no, f, None, 0),
        (F8, F8, Lay::M(8, 16), [256, 512], no, f, None, 0),
        (DS, DS, Lay::C(4), [512, 1152], no, f, p(f, f, 256, 0), 0),
        (NV, NV, Lay::C(3), [640, 640], no, f, None, 1),
        (NV, NV, Lay::C(3), [640, 768], no, f, p(f, f, 128, 3), 3),
        (NV, NV, Lay::C(8), [256, 512], no, f, p(t, f, 64, 1), 0),
        (DS, W4, Lay::C(6), [512, 1024], no, f, p(f, t, 256, 0), 0),
        (BF, BF, Lay::M(6, 1), [384, 4104], no, f, None, 0),
        (NV, NV, Lay::M(12, 40), [256, 2048], no, f, p(t, f, 64, 4), 2),
        (DS, W4, Lay::M(5, 130), [128, 1024], no, f, p(t, f, 160, 2), 0),
        (NV, NV, Lay::C(6), [256, 512], sw, t, p(t, f, 64, 0), 0),
        (DS, DS, Lay::C(6), [256, 512], sw, t, p(f, t, 256, 0), 0),
        (NV, NV, Lay::M(5, 40), [128, 512], clamp, t, p(t, f, 64, 0), 0),
        (BF, W4, Lay::C(6), [256, 576], no, f, p(f, f, 256, 0), 0),
        (BF, NV, Lay::M(5, 72), [128, 512], sw, f, p(t, f, 96, 0), 0),
        (BF, u64_, Lay::M(5, 72), [128, 512], sw, f, p(t, f, 96, 0), 0),
    ];
    cases.iter().for_each(|&case| run(case));
}

#[test]
fn a_dependent_gemm_reads_what_its_producer_landed() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // A chain h = x w1, out = h w2 under `kernel.pdl`, each GEMM the
    // programmatic dependent of the one before.
    let (m, k, n, n2) = (8usize, 8192usize, 1024usize, 512usize);
    let mut b = Bench::new(0xd9);
    let (x, w1, w2) = (b.bf16(m * k).0, b.bf16(n * k).0, b.bf16(n2 * n).0);
    let (h, out, bf) = (b.zeros(m * n * 2), b.zeros(m * n2 * 2), |at| {
        Operand::plain(at, Elem::Bf16)
    });
    let (s1, s2) = (
        [m as u32, n as u32, k as u32],
        [m as u32, n2 as u32, n as u32],
    );
    let unsplit = |s: [u32; 3]| Plan {
        splits: 1,
        ..Plan::choose(s[0], s[1], s[2], 148)
    };
    let chain = |ctx: &Ctx, alpha: f32| {
        gemm_with(ctx, bf(x), bf(w1), h, s1, alpha, unsplit(s1)).expect("the producer fires");
        gemm_with(ctx, bf(h), bf(w2), out, s2, 1.0, unsplit(s2)).expect("the dependent fires");
    };
    chain(&b.ctx(), 1.0);
    let base = b.ctx().tuning();
    let pdl = b.ctx().with_tuning(Tuning { pdl: true, ..base });
    again(&b, &[pdl], (out, m * n2), "the dependent chain", &chain);
}
