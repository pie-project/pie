//! The compressed-attention (pool) family against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::attn::pool;
use kernels_xla::{KvPool, RaggedTensor, Tensor};

const PS: usize = 4;
/// Request 0 holds pages 3, 1, 4; request 1 pages 0, 2.
const PAGES: [&[u32]; 2] = [&[3, 1, 4], &[0, 2]];
const CELLS: usize = 5 * PS;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| round_bf16(self.next() * scale)).collect()
    }
}

fn cell(req: usize, pos: usize) -> usize {
    PAGES[req][pos / PS] as usize * PS + pos % PS
}

fn kv_pool(b: &mut Bench, keys: Tensor) -> KvPool {
    let indices: Vec<u32> = PAGES.iter().flat_map(|p| p.iter().copied()).collect();
    let indptr = [0u32, 3, 5];
    let pi = b.u32(indices.len() as u32, 1, &indices);
    let pp = b.u32(3, 1, &indptr);
    KvPool {
        keys,
        values: keys,
        page_indices: pi,
        page_indptr: pp,
        page_size: PS as i32,
        max_pages: 3,
        seq_stride: u64::from(keys.width),
        head_stride: u64::from(keys.width),
    }
}

#[test]
fn boundaries_mark_the_rows_that_close_a_block() {
    let positions = [0i32, 3, 7, 5, 11, 2, 15];
    let reqs = [0i32, 0, 0, 1, 1, 1, 1];
    let valid = [1u8, 1, 1, 1, 0, 1, 1];
    let ratio = 4;
    for prefill in [false, true] {
        let mut b = Bench::new();
        let p = b.i32(7, 1, &positions);
        let r = b.i32(7, 1, &reqs);
        let v = b.u8(7, 1, &valid);
        let bp = b.zeros(Dtype::I32, 7, 1);
        let br = b.zeros(Dtype::I32, 7, 1);
        let bo = b.zeros(Dtype::I32, 7, 1);
        let ip = b.i32(3, 1, &[0, 3, 7]);
        if !b
            .run(|ctx| {
                if prefill {
                    pool::boundary_prefill(
                        ctx,
                        RaggedTensor {
                            data: p,
                            indptr: ip,
                        },
                        r,
                        v,
                        ratio,
                        bp,
                        br,
                        bo,
                    )
                } else {
                    pool::boundary_decode(ctx, p, r, v, ratio, bp, br, bo)
                }
            })
            .unwrap()
        {
            return;
        }
        let r4 = ratio as i32;
        let is_b: Vec<bool> = (0..7)
            .map(|t| valid[t] != 0 && (positions[t] + 1) % r4 == 0)
            .collect();
        let want_p: Vec<i32> = (0..7)
            .map(|t| if is_b[t] { positions[t] } else { -1 })
            .collect();
        let want_o: Vec<i32> = (0..7)
            .map(|t| if is_b[t] { positions[t] / r4 * r4 } else { 0 })
            .collect();
        assert_eq!(b.read_i32(bp), want_p);
        assert_eq!(b.read_i32(br), reqs.to_vec());
        assert_eq!(b.read_i32(bo), want_o);
    }
}

#[test]
fn the_compressor_files_its_state_and_pools_each_block() {
    let hd = 6usize;
    let mut rng = Rng(3);
    // State write: coff 2 rows, a pitch wider than the row, two dropped rows.
    let width = 2 * hd;
    let pitch = width + 4;
    let rows = 5usize;
    let kv = rng.bf16s(rows * width, 1.0);
    let sc = rng.bf16s(rows * width, 2.0);
    let wpage = [3u32, u32::MAX, 0, 2, 1];
    let woff = [1u32, 0, 3, 9, 0];
    let sk0 = rng.bf16s(CELLS * pitch, 1.0);
    let ss0 = rng.bf16s(CELLS * pitch, 1.0);
    let mut b = Bench::new();
    let kvt = b.bf16(rows as u32, width as u32, &kv);
    let sct = b.bf16(rows as u32, width as u32, &sc);
    let wp = b.u32(rows as u32, 1, &wpage);
    let wo = b.u32(rows as u32, 1, &woff);
    let sk = b.bf16(CELLS as u32, pitch as u32, &sk0);
    let ss = b.bf16(CELLS as u32, pitch as u32, &ss0);
    let keys = b.zeros(Dtype::Bf16, CELLS as u32, hd as u32);
    let pages = kv_pool(&mut b, keys);
    if !b
        .run(|ctx| pool::state_write(ctx, kvt, sct, &pages, wp, wo, hd as u32, 4, sk, ss))
        .unwrap()
    {
        return;
    }
    let (mut wk, mut ws) = (sk0.clone(), ss0.clone());
    for r in 0..rows {
        let (page, off) = (wpage[r] as i32, woff[r] as usize);
        if page < 0 || off >= PS {
            continue;
        }
        let at = (page as usize * PS + off) * pitch;
        wk[at..at + width].copy_from_slice(&kv[r * width..(r + 1) * width]);
        ws[at..at + width].copy_from_slice(&sc[r * width..(r + 1) * width]);
    }
    assert_close(&b.read_f32(sk), &wk, 0.0, 0.0);
    assert_close(&b.read_f32(ss), &ws, 0.0, 0.0);

    // Gather: ratio 4 without ape (coff 2), ratio 3 with a one-head ape.
    let bpos = [7i32, -1, 3, 11, 5];
    let breq = [0i32, 0, 1, 0, 1];
    for (ratio, ape) in [(4usize, false), (3, true)] {
        let coff = if ape { 1 } else { 2 };
        let window = coff * ratio;
        let apev = rng.bf16s(ratio * hd, 1.0);
        let mut b = Bench::new();
        let bp = b.i32(5, 1, &bpos);
        let br = b.i32(5, 1, &breq);
        let sk = b.bf16(CELLS as u32, pitch as u32, &sk0);
        let ss = b.bf16(CELLS as u32, pitch as u32, &ss0);
        let at = b.f32(ratio as u32, hd as u32, &apev);
        let out = b.zeros(Dtype::Bf16, 5, hd as u32);
        let keys = b.zeros(Dtype::Bf16, CELLS as u32, hd as u32);
        let pages = kv_pool(&mut b, keys);
        b.run(|ctx| {
            pool::gather(
                ctx,
                bp,
                br,
                &pages,
                hd as u32,
                ratio as u32,
                sk,
                ss,
                ape.then_some(at),
                out,
            )
        })
        .unwrap();
        let got = b.read_f32(out);
        for r in 0..5 {
            for d in 0..hd {
                let want = if bpos[r] < 0 {
                    0.0
                } else {
                    let mut terms = Vec::new();
                    for i in 0..window {
                        let pos = bpos[r] + i as i32 - (window as i32 - 1);
                        if pos < 0 {
                            continue;
                        }
                        let col = if i >= ratio { hd } else { 0 } + d;
                        let c = cell(breq[r] as usize, pos as usize);
                        let mut s = ss0[c * pitch + col];
                        if ape {
                            s += apev[(pos as usize % ratio) * hd + col];
                        }
                        terms.push((s, sk0[c * pitch + col]));
                    }
                    let m = terms.iter().map(|t| t.0).fold(f32::NEG_INFINITY, f32::max);
                    let z: f32 = terms.iter().map(|t| (t.0 - m).exp()).sum();
                    terms.iter().map(|t| (t.0 - m).exp() * t.1).sum::<f32>() / z
                };
                let g = got[r * hd + d];
                assert!(
                    (g - want).abs() <= 1e-2 + 1e-2 * want.abs(),
                    "ratio {ratio} row {r} lane {d}: {g} vs {want}"
                );
            }
        }
    }
}

#[test]
fn pooled_readers_attend_over_closed_blocks() {
    let (hd, heads, ratio) = (8usize, 3usize, 2usize);
    let mut rng = Rng(9);
    // kv_append: entries land at their boundary cells.
    let entries = rng.bf16s(4 * hd, 1.0);
    let bpos = [1i32, -1, 5, 3];
    let breq = [0i32, 0, 1, 1];
    let keys0 = rng.bf16s(CELLS * hd, 1.0);
    let mut b = Bench::new();
    let et = b.bf16(4, hd as u32, &entries);
    let bp = b.i32(4, 1, &bpos);
    let br = b.i32(4, 1, &breq);
    let keys = b.bf16(CELLS as u32, hd as u32, &keys0);
    let wp = b.u32(4, 1, &[0; 4]);
    let pages = kv_pool(&mut b, keys);
    if !b
        .run(|ctx| pool::kv_append(ctx, et, bp, br, &pages, wp, wp))
        .unwrap()
    {
        return;
    }
    let mut want = keys0.clone();
    for r in 0..4 {
        if bpos[r] >= 0 {
            let c = cell(breq[r] as usize, bpos[r] as usize);
            want[c * hd..(c + 1) * hd].copy_from_slice(&entries[r * hd..(r + 1) * hd]);
        }
    }
    assert_close(&b.read_f32(keys), &want, 0.0, 0.0);

    // Readers: five query rows over the two requests' closed blocks.
    let rows = 5usize;
    let positions = [0i32, 4, 11, 1, 7];
    let reqs = [0i32, 0, 0, 1, 1];
    let q = rng.bf16s(rows * heads * hd, 1.0);
    let top_k = 3usize;
    let selection = [0i32, 2, -1, 1, 5, 0, 3, 4, 1, 0, 0, 0, 2, -1, 1];
    let scale = 0.4f32;
    for selected in [false, true] {
        let mut b = Bench::new();
        let qt = b.bf16(rows as u32, (heads * hd) as u32, &q);
        let pt = b.i32(rows as u32, 1, &positions);
        let rt = b.i32(rows as u32, 1, &reqs);
        let st = b.i32(rows as u32, top_k as u32, &selection);
        let keys = b.bf16(CELLS as u32, hd as u32, &keys0);
        let o = b.zeros(Dtype::Bf16, rows as u32, (heads * hd) as u32);
        let lse = b.zeros(Dtype::F32, rows as u32, heads as u32);
        let pages = kv_pool(&mut b, keys);
        b.run(|ctx| {
            if selected {
                pool::attention_lse_selected(
                    ctx,
                    qt,
                    pt,
                    rt,
                    st,
                    &pages,
                    ratio as u32,
                    top_k as u32,
                    heads as u32,
                    hd as u32,
                    scale,
                    o,
                    lse,
                )
            } else {
                pool::attention_lse(
                    ctx,
                    qt,
                    pt,
                    rt,
                    &pages,
                    ratio as u32,
                    heads as u32,
                    hd as u32,
                    scale,
                    o,
                    lse,
                )
            }
        })
        .unwrap();
        let (go, gl) = (b.read_f32(o), b.read_f32(lse));
        for r in 0..rows {
            let visible = (positions[r] + 1) / ratio as i32;
            let ids: Vec<i32> = if selected {
                selection[r * top_k..(r + 1) * top_k]
                    .iter()
                    .copied()
                    .filter(|&c| c >= 0 && c < visible)
                    .collect()
            } else {
                (0..visible).collect()
            };
            for h in 0..heads {
                let qh = &q[(r * heads + h) * hd..(r * heads + h + 1) * hd];
                let ks: Vec<&[f32]> = ids
                    .iter()
                    .map(|&c| {
                        let at = cell(reqs[r] as usize, (c as usize + 1) * ratio - 1);
                        &keys0[at * hd..(at + 1) * hd]
                    })
                    .collect();
                let s: Vec<f32> = ks
                    .iter()
                    .map(|k| k.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale)
                    .collect();
                let at = (r * heads + h) * hd;
                if s.is_empty() {
                    assert_eq!(gl[r * heads + h], f32::NEG_INFINITY);
                    assert_close(&go[at..at + hd], &vec![0.0; hd], 0.0, 0.0);
                    continue;
                }
                let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = s.iter().map(|v| (v - m).exp()).sum();
                let want: Vec<f32> = (0..hd)
                    .map(|d| {
                        s.iter()
                            .zip(&ks)
                            .map(|(v, k)| (v - m).exp() * k[d])
                            .sum::<f32>()
                            / z
                    })
                    .collect();
                assert_close(&go[at..at + hd], &want, 1e-2, 1e-2);
                let lw = (z.ln() + m) * std::f32::consts::LOG2_E;
                assert_close(&gl[r * heads + h..r * heads + h + 1], &[lw], 1e-4, 1e-4);
            }
        }
    }
}
