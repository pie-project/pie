//! The activation quantizers, the fp8b128 requantize, the routed GLUs and
//! the e4m3 table gathers against the host's rounding of the same values.

use crate::common::linear_ref::{Bench, TOL, check, glu, pick, quant_e4m3, quant_nvfp4, t, tab};
use crate::common::tc5::{e2m1, e4m3, ue8m0};
use crate::common::{from_bf16, to_bf16};
use dtype::Dtype::{Bf16, E4m3, E4m3tensor, E8m0, F32, I32, U8};
use kernels_cuda::linear::act_quant::{Format, Source, quantize, read_back};
use kernels_cuda::linear::gemm_sm100::{Glu, atom_bytes, atom_offset};
use kernels_cuda::linear::glu;

#[test]
fn the_activation_quantizers_land_what_the_host_rounds() {
    let (e, nv) = (|gran| Format::E4m3 { gran }, Format::Nvfp4);
    // (format, rows, k, rows a token): e4m3 in 128s and 32s, a token's
    // codes landed once at each of its routes' rows (every seventh
    // dropped); NVFP4 under one global, and routed under each expert's (a
    // dropped route reading back zeros).
    #[rustfmt::skip]
    let cases = [(e(128), 37, 512, 1), (e(32), 37, 512, 1), (e(128), 296, 512, 8), (e(32), 20, 256, 4),
        (nv, 45, 256, 1), (nv, 15, 128, 3)];
    for (fmt, rows, k, div) in cases {
        let what = format!("{fmt:?} {rows} rows of {k}, {div} a token");
        let mut b = Bench::new(0xa8 ^ rows as u64);
        let (gran, nv, tokens) = (fmt.gran() as usize, fmt == Format::Nvfp4, rows / div);
        let (fan, routed, cols) = (!nv && div > 1, nv && div > 1, k / gran);
        // Each row at its own magnitude.
        let gain = |i: usize| 2f32.powi((i / k % 9) as i32 - 4);
        let x = tab(tokens * k, |i| from_bf16(to_bf16(b.lcg.unit() * gain(i))));
        let pos = tab(rows, |r| {
            pick(r % 7 == 3, -1, ((r * 101 + 11) % rows) as i32)
        });
        let routes = tab(rows, |r| pick(r == 4, -1, ((r * 7 + 1) % 4) as i32));
        let input = [0.01f32, 0.02, 0.005, 0.0371];
        let xt = t(b.up(&tab(x.len(), |i| to_bf16(x[i]))), tokens, k, Bf16);
        let routes_t = t(b.up(&routes), tokens, div, I32);
        let input_scales = nv.then(|| t(b.up(&input), 4, 1, F32));
        let (rows32, div32, routes_t) = (rows as u32, div as u32, routed.then_some(routes_t));
        let pos_at = pick(fan, b.up(&pos), 0);
        #[rustfmt::skip]
        let src = Source { x: xt, rows: rows32, div: div32, routes: routes_t, input_scales, pos: pos_at, m_max: 0 };
        let bytes = atom_bytes(rows, k, gran);
        let (codes, atoms, back) = (b.zeros(rows * k), b.zeros(bytes), b.zeros(rows * k * 2));
        let ctx = b.ctx();
        quantize(&ctx, "test", fmt, src, codes, atoms, bytes).expect("quantizes");
        if !fan {
            read_back(&ctx, "test", fmt, src, back).expect("reads back");
        }
        let (codes, atoms): (Vec<u8>, Vec<u8>) = (b.down(codes, rows * k), b.down(atoms, bytes));
        let (back, mut off) = (b.read(back, rows * k), [0usize; 2]);
        let nib = |j: usize| codes[j / 2] >> (4 * (j % 2)) & 0xf;
        for r in 0..rows {
            let (dest, e) = (pick(fan, pos[r], r as i32), pick(routed, routes[r], 0));
            if dest < 0 || e < 0 {
                let zeros = back[r * k..][..k].iter().all(|&v| v == 0.0);
                assert!(fan || zeros, "{what}: dropped route {r}");
                continue;
            }
            let (dest, gs) = (dest as usize, 1.0 / input[e as usize]);
            for c in 0..cols {
                let block = &x[(r / div) * k + c * gran..][..gran];
                let (want, sf) = pick(nv, quant_nvfp4(block, gs), quant_e4m3(block));
                let (got, at) = (atoms[atom_offset(dest, c, cols)], dest * k + c * gran);
                let code = |i: usize| pick(nv, nib(at + i), codes[at + i]);
                off[0] += usize::from(got != sf);
                let wrong = (0..gran).filter(|&i| code(i) != want[i]).count();
                off[1] += pick(got == sf, wrong, 0);
                // The read-back is the codes under their scale, exactly.
                let scale = pick(nv, e4m3(got), ue8m0(got));
                let value = |i: usize| f64::from(pick(nv, e2m1(code(i)), e4m3(code(i))) * scale);
                let exact = (0..gran).all(|i| fan || back[r * k + c * gran + i] == value(i));
                assert!(exact, "{what}: row {r} block {c} reads back");
            }
        }
        // e4m3 lands the host's codes; NVFP4's approximate reciprocals may
        // tip a tie in 200.
        let slack = pick(nv, rows * k / 200, 0);
        let few = off[0] * gran <= slack && off[1] <= slack;
        assert!(few, "{what}: {off:?} differ");
    }
}

#[test]
fn the_fp8b128_requantize_lands_every_value_and_back() {
    use kernels_cuda::linear::moe::{dequantize_fp8b128, quantize_fp8b128};
    let (rows, k, cols, mut b) = (256usize, 768usize, 6usize, Bench::new(1));
    // A block's largest code magnitude (448; 240; 224, which drops an
    // exponent; 112, which drops two), all zeros, or subnormals alone.
    let block = |at: usize| at / k / 128 * cols + at % k / 128;
    let kind = |at: usize| [0x7e, 0x6f, 0x6e, 0x5e, 0, 0x07][block(at) % 6];
    let scale = |at: usize| 116 + (block(at) % 9) as u8;
    let codes = tab(rows * k, |at| {
        let (r, max) = (((b.lcg.unit() + 1.0) * 127.9) as u8, kind(at));
        // Each block that may reach its bound reaches it once; no zero is -0.
        let bound = at / k % 128 == 5 && at % 128 == 3;
        let c = pick(bound, max, (r & 0x7f) % (max + 1));
        pick(c == 0, 0, c | (r & 0x80))
    });
    let value = |c: u8, e: u8| f64::from(e4m3(c)) * f64::from(ue8m0(e));
    let plane = tab(rows * k, |at| {
        ((value(codes[at], scale(at)) as f32).to_bits() >> 16) as u16
    });
    let bytes = atom_bytes(rows, k, 128);
    let (src, got, atoms) = (b.up(&plane), b.zeros(rows * k), b.zeros(bytes));
    quantize_fp8b128(&b.ctx(), t(src, rows, k, Bf16), got, atoms).expect("requantizes");
    b.sync();
    let (got, atoms): (Vec<u8>, Vec<u8>) = (b.down(got, rows * k), b.down(atoms, bytes));
    // Each value lands exactly, under the least exponent that holds its
    // block: its own, or one or two below where its codes sit low.
    let mut drops = [0usize; 4];
    for at in 0..rows * k {
        let held = atoms[atom_offset(at / k, at % k / 128, cols)];
        let same = value(got[at], held) == value(codes[at], scale(at));
        assert!(same, "value {at}");
        if at % (128 * k) < k && at % 128 == 0 && kind(at) != 0 {
            drops[usize::from(scale(at) - held).min(3)] += 1;
        }
    }
    let all = drops[..3].iter().all(|&d| d > 0);
    assert!(all, "blocks kept, dropped one, dropped two: {drops:?}");
    // And back: the bf16 it was given, bit for bit.
    let (codes, atoms, back) = (b.up(&got), b.up(&atoms), b.zeros(rows * k * 2));
    let (r, k32) = (rows as u32, k as u32);
    dequantize_fp8b128(&b.ctx(), codes, atoms, back, r, k32).expect("dequantizes");
    b.sync();
    let back: Vec<u16> = b.down(back, rows * k);
    assert!(back == plane, "the round trip lands the plane");
}

#[test]
fn the_routed_glus_land_the_clamped_swiglu_and_its_read_back() {
    // DeepSeek-V4's clamped SwiGLU of routed rows, each times its route's
    // weight: over split planes and packed rows, and landing the packed
    // rows' e4m3 read-back (whole 128-column blocks: values past the
    // clamp, and tiny ones near the amax floor).
    #[rustfmt::skip]
    let cases = [(30, 320, 14.0), (48, 2048, 4.0), (5, 384, 40.0), (7, 256, 1e-3)];
    for (rows, inter, spread) in cases {
        let (mut b, limit, n) = (Bench::new(0x5e1 + rows as u64), 10.0, inter as u32);
        let packed = tab(rows * 2 * inter, |_| to_bf16(b.lcg.unit() * spread));
        let weights = tab(rows, |_| b.lcg.unit().abs() + 0.05);
        // Row i's gate column c at `[i][c]`, its up at `[i][inter + c]`.
        let at = |i: usize, o: usize| i / inter * 2 * inter + o + i % inter;
        let half = |o: usize| tab(rows * inter, |i| packed[at(i, o)]);
        let gate = t(b.up(&half(0)), rows, inter, Bf16);
        let up = t(b.up(&half(inter)), rows, inter, Bf16);
        let pt = t(b.up(&packed), rows, 2 * inter, Bf16);
        let wt = t(b.up(&weights), rows, 1, F32);
        let (mut ys, ctx) = ([(); 3].map(|_| b.out(rows, inter)), b.ctx());
        glu::swiglu_clamp_split_routed(&ctx, gate, up, wt, 1, limit, &mut ys[0]).expect("fires");
        glu::swiglu_clamp_routed(&ctx, pt, n, wt, 1, limit, &mut ys[1]).expect("fires");
        let took = glu::swiglu_clamp_routed_e4m3(&ctx, pt, n, wt, 1, limit, &mut ys[2]);
        let whole = took.expect("fires") == (inter % 128 == 0);
        assert!(whole, "{rows}x{inter}: whole blocks take the read-back");
        let v = |i: usize, o: usize| f64::from(from_bf16(packed[at(i, o)]));
        let (f, w) = (Glu::SwigluClamp { limit }, |i: usize| {
            f64::from(weights[i / inter])
        });
        let want = tab(rows * inter, |i| (glu(f, v(i, 0), v(i, inter)) * w(i), 0.0));
        b.check(ys[0].ptr, &want, TOL, &format!("{rows}x{inter} split"));
        b.check(ys[1].ptr, &want, TOL, &format!("{rows}x{inter} packed"));
        if inter % 128 == 0 {
            // The plain rows' read-back, which read back are themselves.
            let fmt = Format::E4m3 { gran: 128 };
            let (back, at_n) = ([(); 2].map(|_| b.zeros(rows * inter * 2)), rows * inter);
            for (y, at) in [ys[1], ys[2]].into_iter().zip(back) {
                read_back(&ctx, "test", fmt, Source::rows(y, None), at).expect("reads back");
            }
            let [fused, back, again] = [ys[2].ptr, back[0], back[1]].map(|at| b.read(at, at_n));
            let same = fused == back && again == fused;
            assert!(same, "{rows}x{inter}: the read-back GLU");
        }
    }
}

#[test]
fn the_alpha_glu_lands_alike_over_packed_rows_and_split_planes() {
    // MiniMax's clamped SwiGLU, `(u + 1) * g * sigmoid(alpha * g)`: its
    // gate and up packed in one row, and held apart where the pack left them.
    for (rows, inter) in [(41, 3072), (5, 200)] {
        let (mut b, limit, alpha) = (Bench::new(0xa1f + rows as u64), 7.0, 1.702);
        let packed = tab(rows * 2 * inter, |_| to_bf16(b.lcg.unit() * 20.0));
        let at = |i: usize, o: usize| i / inter * 2 * inter + o + i % inter;
        let half = |o: usize| tab(rows * inter, |i| packed[at(i, o)]);
        let gate = t(b.up(&half(0)), rows, inter, Bf16);
        let up = t(b.up(&half(inter)), rows, inter, Bf16);
        let pt = t(b.up(&packed), rows, 2 * inter, Bf16);
        let (mut ys, ctx, n) = ([(); 2].map(|_| b.out(rows, inter)), b.ctx(), inter as u32);
        glu::swiglu_clamp_alpha(&ctx, pt, n, 1, limit, alpha, &mut ys[0]).expect("fires");
        glu::swiglu_clamp_alpha_split(&ctx, gate, up, 1, limit, alpha, &mut ys[1]).expect("fires");
        let v = |i: usize, o: usize| f64::from(from_bf16(packed[at(i, o)]));
        let f = Glu::SwigluClampAlpha { limit, alpha };
        let want = tab(rows * inter, |i| (glu(f, v(i, 0), v(i, inter)), 0.0));
        b.check(ys[0].ptr, &want, TOL, &format!("{rows}x{inter} packed"));
        b.check(ys[1].ptr, &want, TOL, &format!("{rows}x{inter} split"));
    }
}

#[test]
fn the_e4m3_table_gathers_widen_only_their_rows() {
    use kernels_cuda::gather::{embed_concat_e4m3, embed_concat_mxfp8};
    // e4m3 rows under one scale, 16-byte runs a thread (160) and a code a
    // thread (24); DeepSeek-V4.1's Engram MXFP8 rows (256, eight exponents).
    for (width, mx) in [(160, false), (24, false), (256, true)] {
        let (vocab, heads, rows, scale) = (777usize, 16usize, 5usize, 0.0123f32);
        let mut b = Bench::new(0x8f8 + width as u64);
        let table = tab(vocab * width, |_| {
            ((b.lcg.unit() + 1.0) * 127.5) as u8 & 0xf7
        });
        let exps = tab(vocab * width / 32, |i| 112 + (i * 13 % 24) as u8);
        // An id outside the table lands zeros.
        let ids = tab(rows * heads, |i| {
            pick(i % 13 == 4, -1, ((i * 89 + 7) % 778) as i32)
        });
        let (ids_t, codes) = (t(b.up(&ids), rows, heads, I32), b.up(&table));
        let (mut y, ctx, v) = (b.out(rows, heads * width), b.ctx(), vocab as u32);
        if mx {
            let exps = t(b.up(&exps), vocab, width / 32, E8m0);
            embed_concat_mxfp8(&ctx, ids_t, t(codes, vocab, width, E4m3), exps, v, &mut y)
        } else {
            let (codes, scale) = (t(codes, vocab, width, E4m3tensor), b.up(&[scale]));
            embed_concat_e4m3(&ctx, ids_t, codes, t(scale, 1, 4, U8), v, &mut y)
        }
        .expect("the gather fires");
        let at = |i: usize| ids[i / width].clamp(0, 776) as usize * width + i % width;
        let s = |i: usize| pick(mx, ue8m0(exps[at(i) / 32]), scale);
        let v = |i: usize| f64::from(from_bf16(to_bf16(e4m3(table[at(i)]) * s(i))));
        let live = |i: usize| (0..777).contains(&ids[i / width]);
        let want = tab(rows * heads * width, |i| (pick(live(i), v(i), 0.0), 0.0));
        let got = b.read(y.ptr, want.len());
        check(&got, &want, 0.0, &format!("width {width}"));
    }
}
