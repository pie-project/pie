//! The routed selects (bf16, affine, MXFP4, NVFP4, and the W4A8 and W4A4
//! ones that quantize their rows) and the routed MLP kept by expert,
//! against the host's dot of each route's rows and its expert's weight.

use crate::common::linear_ref::{Bench, TOL, affine, check, distance, dot, fp8_bank};
use crate::common::linear_ref::{glu_want, mlp, pick, t, tab, values};
use crate::common::tc5::ue8m0;
use crate::common::{SM100, TC5, from_bf16, to_bf16};
use dtype::Dtype::{Bf16, E4m3, F32, I32, Nvfp4, U8};
use kernels_cuda::linear::act_quant::{AlphaFrom, Bank, Format, Source, read_back, select};
use kernels_cuda::linear::moe::{self, ExpertTable, GluBank, GroupSeat};
use kernels_cuda::linear::nvfp4;
use kernels_cuda::linear::tc5::{Elem, Glu, supported};
use kernels_cuda::tuning::Tuning;

const RES: ExpertTable = ExpertTable::RESIDENT;
const SEAT: GroupSeat = GroupSeat::RESIDENT;
#[rustfmt::skip]
const W4A8: Tuning = Tuning { moe_w4a8_fused_from: 1, ..SM100 };

/// A select's bank and arm: bf16, affine u4, MXFP4 (biased), NVFP4, MXFP4
/// under atom-laid scales after a W4A8 read-back, and the W4A8 and W4A4
/// selects that quantize their rows.
#[derive(Clone, Copy, Debug, PartialEq)]
#[rustfmt::skip]
enum K { Bf16, Affine, Mx, MxBias, Nv, Atoms, A8, A4 }

/// The act_quant bank of `codes` and `atoms`, `[e][n][k]` MXFP4, or NVFP4
/// under an expert's `alpha` (where it lands, input and tensor scales).
fn bank(codes: u64, atoms: u64, [e, n, k]: [usize; 3], alpha: Option<[u64; 3]>) -> Bank {
    let (experts, n, k, elem) = (e as u32, n as u32, k as u32, Elem::E2m1);
    let [alpha_e, input_scales, tensor_scales] = alpha.unwrap_or_default();
    #[rustfmt::skip]
    let from = AlphaFrom { input_scales, tensor_scales, rows: n };
    let (gran, alpha_from) = (pick(alpha.is_some(), 16, 32), alpha.map(|_| from));
    #[rustfmt::skip]
    let bank = Bank { codes, atoms, experts, n, k, elem, gran, alpha_e, alpha_from };
    bank
}

/// One case: the arm, its tuning, `[experts, tokens, top, n, k]`, whether
/// a route reads its token's row, the GLU, and flags: 1 a live window, 2
/// scales past f16's range, 4 activations past it, 8 the route order kept
/// for a select after.
type Case = (K, Tuning, [usize; 5], bool, Option<Glu>, u8);

fn routed((kind, tuning, [ex, tokens, top, n, k], by_token, f, flags): Case) {
    let what = format!("{kind:?} {ex}x{tokens}x{top} {by_token} {n}x{k} {f:?} {flags}");
    let mut b = Bench::new(0x5e1 ^ (tokens * 7 + n) as u64);
    let (count, quant) = (tokens * top, matches!(kind, K::A8 | K::A4));
    let mut routes = tab(count, |r| ((r / top * 3 + r % top * 5) % ex) as i32);
    // A quantizing select drops a route, and those past a live window.
    let live = pick(flags & 1 == 0, count, tokens * 2 / 5 * top);
    if quant {
        routes[1] = -1;
        routes[live..].fill(0x3fff_ffff);
    }
    let (rows_x, div) = pick(by_token, (tokens, top), (count, 1));
    let (raw, _) = b.lcg.row(rows_x * k);
    let big = |i: usize| flags & 4 != 0 && i % 97 == 5;
    let raw = tab(raw.len(), |i| pick(big(i), to_bf16(65536.0), raw[i]));
    let mut a = tab(raw.len(), |i| f64::from(from_bf16(raw[i])));
    // The bank: `rows` rows an expert, NVFP4's tensor scale a row (an
    // expert's), W4A4's input scale an expert.
    let (rows, nv) = (n << u8::from(f.is_some()), matches!(kind, K::Nv | K::A4));
    let (gran, mut fz) = (pick(nv, 16, 32), (0, 0));
    let mut s = match kind {
        K::Bf16 => b.side(Elem::Bf16, None, ex * rows, k),
        K::Affine => {
            let (s, fs, zs) = affine(&mut b.lcg, [ex * rows, k, 64, 4]);
            fz = (b.up(&fs), b.up(&zs));
            s
        }
        _ => b.side(Elem::E2m1, Some(gran), ex * rows, k),
    };
    let ts = tab(ex * rows, |r| pick(nv, 0.5 + (r / rows) as f32 / 8.0, 1.0));
    let v = values(&s, k);
    let mut w = tab(v.len(), |i| v[i] * ts[i / k]);
    // One row in 40 under scales far outside f16's range.
    let wide = |i: usize| flags & 2 != 0 && i / (k / 32) % 40 == 3;
    for i in (0..s.codes.len()).filter(|&i| wide(i)) {
        let to = 100 + (i / (k / 32) % 51) as u8;
        let ratio = ue8m0(to) / ue8m0(s.codes[i]);
        w[i * 32..][..32].iter_mut().for_each(|v| *v *= ratio);
        s.codes[i] = to;
    }
    let input = tab(ex, |e| 0.004 + 0.003 * e as f32);
    let ((bias, bv), x) = (b.bf16(ex * rows), t(b.up(&raw), rows_x, k, Bf16));
    let atoms = s.scales.as_ref().map_or(&s.codes, |x| &x.0);
    let (c, atoms, sc) = (b.up(&s.bytes), b.up(atoms), b.up(&s.codes));
    let (tsv, inputs) = (b.up(&ts), b.up(&input));
    let (cw, sw) = (rows * k / 2, rows * k / gran);
    let (mx, sc) = (t(c, ex, cw, U8), t(sc, ex, sw, U8));
    let (nvc, nvt) = (t(c, ex, cw, Nvfp4), t(tsv, ex, rows * 4, U8));
    let ctx = b.ctx().with_tuning(tuning);
    let r = t(b.up(&routes), tokens, top, I32);
    let mut y = t(b.up(&vec![0x7f7fu16; count * rows]), count, n, Bf16);
    let back = b.zeros(count * k * 2);
    let fmt = pick(nv, Format::Nvfp4, Format::E4m3 { gran: 128 });
    if flags & 1 != 0 {
        ctx.arm_stage(b.up(&[(live / top) as u32, 0, 0, 0]));
    }
    let took = |took: bool| assert!(took, "the arm takes the fire");
    match (kind, f) {
        (_, Some(g)) => {
            let bias = (kind == K::MxBias).then(|| t(bias, ex, rows, Bf16));
            #[rustfmt::skip]
            let bank = match kind {
                K::Bf16 => GluBank::Bf16(t(c, ex, rows * k, Bf16)),
                K::Nv => GluBank::Nvfp4 { codes: nvc, scales: sc, tensor_scales: nvt },
                _ => GluBank::Mxfp4 { codes: mx, scales: sc, bias },
            };
            moe::select_glu(&ctx, x, bank, r, &mut y, g, RES).map(took)
        }
        (K::Bf16, _) => moe::matmul_select(&ctx, x, t(c, ex, n * k, Bf16), r, &mut y, RES),
        (K::Affine, _) => {
            let (fs, zs) = (t(fz.0, ex, n * k / 32, U8), t(fz.1, ex, n * k / 32, U8));
            moe::matmul_select_quant(&ctx, x, mx, fs, Some(zs), r, &mut y, SEAT)
        }
        (K::Mx, _) => moe::matmul_select_quant(&ctx, x, mx, sc, None, r, &mut y, SEAT),
        (K::MxBias, _) => {
            let bias = t(bias, ex, n, Bf16);
            moe::matmul_select_bias(&ctx, x, mx, sc, bias, r, &mut y, SEAT)
        }
        (K::Nv, _) => nvfp4::matmul_select(&ctx, x, nvc, sc, nvt, r, &mut y, SEAT),
        (K::Atoms, _) => {
            let xb = read_back(&ctx, "test", fmt, Source::rows(x, None), back).expect("reads");
            a = b.read(back, rows_x * k);
            let mut go = || moe::select_mxfp4_atoms(&ctx, xb, mx, atoms, r, &mut y);
            let keep = flags & 8 != 0;
            (if keep { moe::keeping_order(go) } else { go() }).map(took)
        }
        _ => {
            let globals = nv.then(|| t(inputs, ex, 1, F32));
            let alpha = nv.then(|| [b.zeros(ex * 4), inputs, tsv]);
            let bank = bank(c, atoms, [ex, n, k], alpha);
            let fired = select(&ctx, "test", fmt, x, r, globals, bank, &mut y);
            ctx.disarm_stage();
            // Each route's rows as the select reads them back.
            let (rows, div) = (live as u32, div as u32);
            let routes = nv.then(|| t(r.ptr, live / top, top, I32));
            #[rustfmt::skip]
            let src = Source { x, rows, div, routes, input_scales: globals, pos: 0, m_max: 0 };
            read_back(&ctx, "test", fmt, src, back).expect("reads back");
            a = b.read(back, live * k);
            fired
        }
    }
    .expect("the select fires");
    ctx.disarm_stage();
    // Each live route's dot (a quantizing one's of its read-back rows, W4A4's
    // times its expert's input scale), through the bias and the GLU.
    let a_of = |r: usize, i: usize| a[pick(quant, r, r / div) * k + i];
    let e_of = |r: usize| routes[r].clamp(0, ex as i32 - 1) as usize;
    let gain = |r: usize| pick(kind == K::A4, f64::from(input[e_of(r)]), 1.0);
    let w_of = |r: usize, c: usize, i: usize| f64::from(w[(e_of(r) * rows + c) * k + i]) * gain(r);
    let d = dot([live, rows, k], a_of, w_of);
    let bias = |r: usize, c: usize| pick(kind == K::MxBias, bv[e_of(r) * rows + c], 0.0);
    let v = |r: usize, c: usize| (d[r * rows + c].0 + bias(r, c), d[r * rows + c].1);
    let glu = |r: usize, c: usize| f.map_or_else(|| v(r, c), |g| glu_want(g, v(r, c), v(r, n + c)));
    let want_at = |i: usize| pick(routes[i / n] < 0, (0.0, 0.0), glu(i / n, i % n));
    let (got, want) = (b.read(y.ptr, count * n), tab(live * n, want_at));
    let tol = pick(flags & 4 != 0, 4.0 * TOL, TOL);
    check(&got[..live * n], &want, tol, &what);
    let sentinel = f64::from(from_bf16(0x7f7f));
    let untouched = got[live * n..].iter().all(|&v| v == sentinel);
    assert!(untouched, "{what}: past the window");
    if flags & 8 != 0 {
        // The row-major select after takes the kept route order.
        let x = pick(kind == K::Atoms, t(back, rows_x, k, Bf16), x);
        let mut y = b.out(count, rows);
        moe::matmul_select_quant(&ctx, x, mx, sc, None, r, &mut y, SEAT).expect("fires");
        b.check(y.ptr, &d, TOL, &format!("{what}: the select after"));
    }
}

#[test]
fn the_routed_selects_answer_the_routed_dot() {
    let d = Tuning::DEFAULT;
    #[rustfmt::skip]
    let (gemv, fp32) = (Tuning { moe_grouped: false, ..d }, Tuning { mxfp4_wmma: false, ..d });
    #[rustfmt::skip]
    let mma = Tuning { mxfp4_mma_until: 65_536, mxfp4_mma_from: 1, ..d };
    #[rustfmt::skip]
    let cvt = Tuning { mxfp4_cvt: true, mxfp4_a8_until: 65_536, mxfp4_cvt_ksplit_until: 6, ..mma };
    #[rustfmt::skip]
    let nv_mma = Tuning { nvfp4_mma_until: 65_536, mxfp4_cvt: true, ..TC5 };
    let (sw, f, t) = (Some(Glu::Swiglu), false, true);
    #[rustfmt::skip]
    let alpha = Some(Glu::SwigluClampAlpha { limit: 1e4, alpha: 1.702 });
    #[rustfmt::skip]
    let mut cases: Vec<Case> = vec![
        // bf16: the GEMV by token and by route (past grid.y's 65535 routes),
        // the grouped cuBLAS leg, tcgen05 and its GLU.
        (K::Bf16, d, [8, 1, 2, 96, 64], t, None, 0),
        (K::Bf16, gemv, [8, 8192, 8, 32, 32], f, None, 0),
        (K::Bf16, d, [8, 48, 2, 96, 64], f, None, 0),
        (K::Bf16, TC5, [16, 70, 4, 192, 256], t, None, 0),
        (K::Bf16, TC5, [16, 48, 4, 128, 256], t, sw, 0),
        // Affine u4: per route, grouped, tcgen05.
        (K::Affine, d, [16, 1, 8, 1408, 2816], t, None, 0),
        (K::Affine, d, [16, 3, 8, 704, 704], f, None, 0),
        (K::Affine, TC5, [16, 40, 8, 256, 704], f, None, 0),
        // MXFP4: per route, grouped WMMA and fp32, tcgen05; the register-
        // dequant kernel at one, two and four 8-route blocks, its GLU
        // keeping the order for the select after; the e2m1 converts (a K
        // split at one token, scales and rows past f16's range); B200's.
        (K::Mx, d, [8, 3, 4, 96, 2240], t, None, 0),
        (K::MxBias, d, [8, 64, 4, 256, 256], t, None, 0),
        (K::MxBias, fp32, [8, 48, 4, 200, 160], t, None, 0),
        (K::MxBias, TC5, [8, 40, 4, 576, 2880], t, None, 0),
        (K::MxBias, mma, [8, 1, 4, 200, 2880], t, None, 0),
        (K::Mx, mma, [8, 12, 4, 384, 2880], t, None, 0),
        (K::Mx, mma, [8, 100, 4, 384, 2880], f, None, 0),
        (K::MxBias, mma, [8, 16, 4, 192, 2880], t, alpha, 8),
        // Past 32 routes an expert, the shared-memory-staged kernel: by
        // token and by route, its GLU under scales past f16's range, a live
        // window, NVFP4 and its GLU.
        (K::MxBias, mma, [8, 96, 4, 256, 2880], t, None, 0),
        (K::Mx, mma, [4, 200, 2, 384, 1024], f, None, 0),
        (K::MxBias, mma, [8, 80, 4, 192, 2880], t, alpha, 2),
        (K::Mx, mma, [8, 120, 4, 256, 512], t, None, 1),
        (K::Mx, cvt, [8, 1, 4, 200, 2880], t, None, 0),
        (K::MxBias, cvt, [8, 40, 4, 576, 2880], t, None, 0),
        (K::MxBias, cvt, [8, 3, 4, 576, 2880], f, None, 2),
        (K::MxBias, cvt, [8, 1, 4, 192, 2880], t, alpha, 2),
        (K::Mx, cvt, [8, 12, 4, 384, 2880], t, None, 4),
        (K::MxBias, SM100, [8, 16, 4, 192, 2880], t, alpha, 0),
        // NVFP4: by route, grouped, tcgen05, the register-dequant kernel and
        // its GLU.
        (K::Nv, d, [16, 3, 10, 640, 2560], t, None, 0),
        (K::Nv, d, [16, 48, 10, 200, 640], f, None, 0),
        (K::Nv, TC5, [16, 48, 10, 128, 640], t, None, 0),
        (K::Nv, nv_mma, [16, 2, 10, 256, 640], f, None, 0),
        (K::Nv, nv_mma, [16, 48, 10, 320, 2560], t, sw, 0),
        (K::Nv, nv_mma, [16, 160, 10, 256, 640], t, None, 0),
        (K::Nv, nv_mma, [16, 120, 10, 320, 2560], t, sw, 0),
    ];
    if supported() {
        // The atom-laid select on the e2m1 converts, keeping its order; the
        // quantizing selects masked (a few routes) and contiguous, by token
        // and by route, under a live window; V4-Flash's leg.
        #[rustfmt::skip]
        cases.extend([(K::Atoms, cvt, [8, 12, 4, 384, 4096], t, None, 8),
            (K::Atoms, cvt, [8, 1, 4, 384, 4096], t, None, 0), (K::A8, d, [16, 1, 6, 2048, 4096], t, None, 0)]);
        for kind in [K::A8, K::A4] {
            #[rustfmt::skip]
            cases.extend([(kind, d, [8, 3, 3, 256, 512], t, None, 0), (kind, d, [8, 40, 3, 256, 512], t, None, 1),
                (kind, d, [8, 70, 3, 256, 512], f, None, 0), (kind, d, [8, 5, 3, 256, 512], f, None, 1)]);
        }
    }
    // The tcgen05 arms are Blackwell's; elsewhere their cases have no arm.
    cases.retain(|case| supported() || !(case.1 == TC5 || case.1 == SM100));
    cases.into_iter().for_each(routed);
}

/// A routed MLP's bank `[experts][rows][k]` of bf16, MXFP4, or against
/// quantized rows NVFP4 (W4A4, `a`: 4), MXFP4 (W4A8, 8) or block FP8 (W8A8,
/// 16), and its values.
fn mlp_bank(b: &mut Bench, a: u8, [ex, rows, k]: [usize; 3]) -> (GluBank, Vec<f32>) {
    let plane = |b: &mut Bench, v: &[u8], w: usize| t(b.up(v), ex, w, U8);
    if a == 16 {
        let (codes, atoms, w) = fp8_bank(b, ex * rows, k, 128, 0x71 ^ k as u64);
        let codes = t(codes, ex, rows * k, E4m3);
        let atoms = t(atoms, ex, rows * k / 128, U8);
        return (GluBank::Fp8A8 { codes, atoms }, w);
    }
    let gran = pick(a == 4, 16, 32);
    let elem = pick(a == 0, Elem::Bf16, Elem::E2m1);
    let s = b.side(elem, (a != 0).then_some(gran), ex * rows, k);
    let (codes, w) = (plane(b, &s.bytes, rows * k / 2), values(&s, k));
    let atoms = s.scales.as_ref().map_or(&s.codes, |x| &x.0);
    let width = rows * k / gran;
    let (atoms, scales) = (plane(b, atoms, width), plane(b, &s.codes, width));
    // W4A4's tensor scale an expert's rows, and input scale an expert.
    let ts = tab(ex * rows, |r| 0.01 + (r / rows) as f32 / 1024.0);
    let input = tab(ex, |e| 0.004 + 0.003 * e as f32);
    let (tensor_scales, input_scales) = (t(b.up(&ts), ex, rows, F32), t(b.up(&input), ex, 1, F32));
    #[rustfmt::skip]
    let bank = match a {
        0 => GluBank::Bf16(t(b.up(&s.bytes), ex, rows * k, Bf16)),
        1 => GluBank::Mxfp4 { codes, scales, bias: None },
        8 => GluBank::Mxfp4A8 { codes, atoms },
        _ => GluBank::Nvfp4A4 { codes, atoms, tensor_scales, input_scales },
    };
    (bank, tab(w.len(), |i| w[i] * pick(a == 4, ts[i / k], 1.0)))
}

#[test]
fn the_routed_mlp_kept_by_expert_answers_the_host() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let (sw, clamp) = (Glu::Swiglu, Glu::SwigluClamp { limit: 7.0 });
    // (rows: 0 bf16, 1 MXFP4, 4 W4A4, 8 W4A8, 16 W8A8; tokens, [hidden,
    // inter, experts, top], GLU, the gate (or DeepSeek's weights first and
    // the shared row added), a live window, the distance its quantized rows
    // keep): masked and contiguous, the gate-ups swapped and not.
    #[rustfmt::skip]
    let cases = [
        (0, 64, [256, 128, 16, 4], sw, Some(false), Some((40, 8)), 0.01),
        (1, 300, [512, 128, 16, 4], sw, Some(false), None, 0.01),
        (1, 64, [256, 128, 16, 4], sw, None, Some((17, 3)), 0.01),
        (4, 64, [512, 256, 64, 10], sw, None, None, 0.3),
        (4, 8, [512, 256, 16, 8], sw, None, None, 0.3),
        (4, 300, [256, 128, 16, 4], sw, Some(false), None, 0.3),
        (8, 64, [512, 256, 64, 10], Glu::Situ { beta: 4.0, cap: 25.0 }, None, None, 0.25),
        (8, 300, [256, 128, 16, 4], clamp, Some(false), None, 0.25),
        (8, 64, [512, 256, 64, 6], clamp, Some(true), None, 0.25),
        (16, 160, [512, 256, 16, 4], sw, None, None, 0.1),
    ];
    for (a, tokens, [hidden, inter, ex, top], g, gate, window, tol) in cases {
        let what = format!("{a} {tokens}x{top} over {ex}, gate {gate:?}, window {window:?}");
        let mut b = Bench::new(0x3e ^ tokens as u64);
        let routes = crate::common::host::routes_of(tokens, ex, top);
        let weights = tab(routes.len(), |i| 0.1 + (i % 7) as f32 * 0.05);
        let ((x, xv), (sh, sv)) = (b.bf16(tokens * hidden), b.bf16(tokens * hidden));
        let (gates, gv) = b.bf16(tokens);
        // The bf16 and narrow W4A4 fires norm their routes under a scale
        // (Muse Spark's).
        let ((scale, scv), normed) = (b.bf16(hidden), tokens < 64 || a == 0);
        let norm = normed.then(|| (t(scale, 1, hidden, Bf16), 1e-6));
        let (up, uw) = mlp_bank(&mut b, a, [ex, 2 * inter, hidden]);
        let (down, dw) = mlp_bank(&mut b, a, [ex, hidden, inter]);
        let (x, it) = (t(x, tokens, hidden, Bf16), inter as u32);
        let mut y = b.out(tokens, hidden);
        let r = t(b.up(&routes), tokens, top, I32);
        let wt = t(b.up(&weights), tokens, top, F32);
        let ctx = b.ctx().with_tuning(pick(a < 4, TC5, W4A8));
        if let Some((live, first)) = window {
            ctx.arm_stage(b.up(&[live, first]));
        }
        let (st, gt) = (t(sh, tokens, hidden, Bf16), t(gates, tokens, 1, Bf16));
        let m = moe::RoutedMlp {
            x,
            up,
            glu: g,
            down,
            routes: r,
            weights: wt,
            inter: it,
        };
        #[rustfmt::skip]
        let took = match gate {
            Some(true) => moe::experts_weighted_first(&ctx, m, st, &mut y),
            _ => moe::experts(&ctx, m, &mut y, RES, RES, gate.map(|_| (st, gt)), norm),
        };
        assert!(took.expect("fires"), "{what}: the grouped GEMM takes it");
        let upv = |e: usize, o: usize, i: usize| f64::from(uw[(e * 2 * inter + o) * hidden + i]);
        let downv = |e: usize, o: usize, c: usize| f64::from(dw[(e * hidden + o) * inter + c]);
        let shared = gate.map(|f| (&sv[..], (!f).then_some(&gv[..])));
        let first = gate == Some(true);
        #[rustfmt::skip]
        let want = mlp(&xv, (&routes, &weights), [hidden, inter, top], upv, downv, g, first, shared, normed.then_some(&scv[..]));
        let (lo, hi) = window.map_or((0, tokens), |(l, f)| (f as usize, (f + l) as usize));
        let (got, span) = (b.read(y.ptr, tokens * hidden), lo * hidden..hi * hidden);
        // Quantized rows answer to their codes' rounding on both legs.
        let far = distance(&got[span.clone()], &want[span]);
        assert!(far < tol, "{what}: {far:.4} from the host's product");
    }
}

#[test]
fn the_w8a16_selects_answer_the_routed_dot() {
    use kernels_cuda::linear::moe::select_fp8_w8a16;
    // (experts, tokens, top_k, n, k, by_token): a decode's few routes an
    // expert, a slice of 16 and of 32, and the down select's routes as rows.
    #[rustfmt::skip]
    let cases = [(8, 1, 4, 256, 512, true), (8, 3, 2, 136, 1024, true), (4, 20, 2, 512, 256, true),
        (16, 64, 8, 128, 512, false), (8, 7, 4, 384, 768, false)];
    for (ex, tokens, top, n, k, by_token) in cases {
        let what = format!("W8A16 select {ex}x{tokens}x{top} {n}x{k} by_token={by_token}");
        let mut b = Bench::new(0x8e1 ^ (tokens * 13 + n + k) as u64);
        let count = tokens * top;
        let routes = tab(count, |r| ((r / top * 3 + r % top * 5) % ex) as i32);
        let (rows_x, div) = pick(by_token, (tokens, top), (count, 1));
        let (x, a) = b.bf16(rows_x * k);
        let s = b.side(Elem::E4m3, Some(128), ex * n, k);
        let atoms = &s.scales.as_ref().expect("scaled").0;
        let (codes, atoms) = (b.up(&s.bytes), b.up(atoms));
        let r = t(b.up(&routes), tokens, top, I32);
        let mut y = b.out(count, n);
        let ctx = b.ctx();
        let x = t(x, rows_x, k, Bf16);
        select_fp8_w8a16(
            &ctx, "test", x, codes, atoms, ex as u32, 128, r, &mut y, None,
        )
        .expect("the W8A16 select fires");
        let w = |r: usize, c: usize, i: usize| f64::from(s.value(routes[r] as usize * n + c, k, i));
        let want = dot([count, n, k], |r, i| a[r / div * k + i], w);
        b.check(y.ptr, &want, TOL, &what);
    }
}

