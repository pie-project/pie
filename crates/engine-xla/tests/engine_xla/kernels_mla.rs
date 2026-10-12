//! The MLA family against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::attn::mla;
use kernels_xla::{KvPool, RaggedTensor};

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

#[test]
fn latents_norm_split_and_rotate() {
    let mut rng = Rng(1);
    let (rows, rank, rope, extra) = (5usize, 96usize, 32usize, 8usize);
    let src = rank + rope + extra;
    let kv_a = rng.bf16s(rows * src, 2.0);
    let w = rng.bf16s(rank, 1.0);
    let pos = [0i32, 3, 17, 250, 4096];
    let theta = 10000.0f32;
    let mut b = Bench::new();
    let at = b.bf16(rows as u32, src as u32, &kv_a);
    let wt = b.bf16(1, rank as u32, &w);
    let pt = b.i32(rows as u32, 1, &pos);
    let c1 = b.zeros(Dtype::Bf16, rows as u32, rank as u32);
    let p1 = b.zeros(Dtype::Bf16, rows as u32, rope as u32);
    let c2 = b.zeros(Dtype::Bf16, rows as u32, rank as u32);
    let p2 = b.zeros(Dtype::Bf16, rows as u32, rope as u32);
    if !b
        .run(|ctx| {
            mla::latents(ctx, at, wt, 1e-6, rank as u32, c1, p1)?;
            mla::latents_rope(
                ctx,
                at,
                pt,
                wt,
                1e-6,
                rank as u32,
                rope as u32,
                theta,
                c2,
                p2,
            )
        })
        .unwrap()
    {
        return;
    }
    let mut want_c = vec![0.0; rows * rank];
    let mut want_p = vec![0.0; rows * rope];
    let mut want_r = vec![0.0; rows * rope];
    for r in 0..rows {
        let row = &kv_a[r * src..(r + 1) * src];
        let ms: f64 = row[..rank]
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            / rank as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for i in 0..rank {
            want_c[r * rank + i] = round_bf16((f64::from(row[i]) * inv * f64::from(w[i])) as f32);
        }
        let pe = &row[rank..rank + rope];
        want_p[r * rope..(r + 1) * rope].copy_from_slice(pe);
        let half = rope / 2;
        for i in 0..half {
            let ang = f64::from(pos[r]) * f64::from(theta).powf(-2.0 * i as f64 / rope as f64);
            let (s, c) = ang.sin_cos();
            let (x1, x2) = (f64::from(pe[i]), f64::from(pe[i + half]));
            want_r[r * rope + i] = round_bf16((x1 * c - x2 * s) as f32);
            want_r[r * rope + i + half] = round_bf16((x1 * s + x2 * c) as f32);
        }
    }
    assert_close(&b.read_f32(c1), &want_c, 1e-3, 8e-3);
    assert_close(&b.read_f32(c2), &want_c, 1e-3, 8e-3);
    assert_close(&b.read_f32(p1), &want_p, 0.0, 0.0);
    assert_close(&b.read_f32(p2), &want_r, 2e-3, 8e-3);
}

#[test]
fn q_b_splits_and_kv_b_absorbs_both_ways() {
    let mut rng = Rng(2);
    let (t, heads, nope, rope, rank, vd) = (6usize, 3usize, 24usize, 8usize, 40usize, 20usize);
    let q_b = rng.bf16s(t * heads * (nope + rope), 1.0);
    let kv_b = rng.bf16s(heads * (nope + vd) * rank, 1.0);
    let latent = rng.bf16s(t * heads * rank, 1.0);
    let mut b = Bench::new();
    let qb = b.bf16(t as u32, (heads * (nope + rope)) as u32, &q_b);
    let kb = b.bf16((heads * (nope + vd)) as u32, rank as u32, &kv_b);
    let lt = b.bf16(t as u32, (heads * rank) as u32, &latent);
    let qn = b.zeros(Dtype::Bf16, t as u32, (heads * nope) as u32);
    let qp = b.zeros(Dtype::Bf16, t as u32, (heads * rope) as u32);
    let ql = b.zeros(Dtype::Bf16, t as u32, (heads * rank) as u32);
    let o = b.zeros(Dtype::Bf16, t as u32, (heads * vd) as u32);
    let (h, r) = (heads as u32, rank as u32);
    if !b
        .run(|ctx| {
            mla::split_q_b(ctx, qb, h, nope as u32, rope as u32, qn, qp)?;
            mla::absorb_q(ctx, qn, kb, h, r, nope as u32, vd as u32, ql)?;
            mla::absorb_out(ctx, lt, kb, h, r, vd as u32, nope as u32, o)
        })
        .unwrap()
    {
        return;
    }
    let per = nope + rope;
    let mut want_n = vec![0.0; t * heads * nope];
    let mut want_p = vec![0.0; t * heads * rope];
    for i in 0..t * heads {
        want_n[i * nope..(i + 1) * nope].copy_from_slice(&q_b[i * per..i * per + nope]);
        want_p[i * rope..(i + 1) * rope].copy_from_slice(&q_b[i * per + nope..(i + 1) * per]);
    }
    assert_close(&b.read_f32(qn), &want_n, 0.0, 0.0);
    assert_close(&b.read_f32(qp), &want_p, 0.0, 0.0);
    let wk = |hh: usize, j: usize, i: usize| f64::from(kv_b[(hh * (nope + vd) + j) * rank + i]);
    let mut want_l = vec![0.0; t * heads * rank];
    let mut want_o = vec![0.0; t * heads * vd];
    for tt in 0..t {
        for hh in 0..heads {
            for i in 0..rank {
                let s: f64 = (0..nope)
                    .map(|j| f64::from(want_n[(tt * heads + hh) * nope + j]) * wk(hh, j, i))
                    .sum();
                want_l[(tt * heads + hh) * rank + i] = round_bf16(s as f32);
            }
            for j in 0..vd {
                let s: f64 = (0..rank)
                    .map(|i| f64::from(latent[(tt * heads + hh) * rank + i]) * wk(hh, nope + j, i))
                    .sum();
                want_o[(tt * heads + hh) * vd + j] = round_bf16(s as f32);
            }
        }
    }
    assert_close(&b.read_f32(ql), &want_l, 1e-2, 8e-3);
    assert_close(&b.read_f32(o), &want_o, 1e-2, 8e-3);
}

/// An MLA pool: latent and rope planes, and each lane's pages.
struct Pool {
    ps: usize,
    rank: usize,
    rope: usize,
    lanes: Vec<Vec<i32>>,
    ckv: Vec<f32>,
    kpe: Vec<f32>,
    slots: usize,
    max_pages: u32,
}

impl Pool {
    fn slot(&self, lane: usize, kp: usize) -> usize {
        self.lanes[lane][kp / self.ps] as usize * self.ps + kp % self.ps
    }

    fn bind(&self, b: &mut Bench) -> KvPool {
        let keys = b.bf16(self.slots as u32, self.rank as u32, &self.ckv);
        let values = b.bf16(self.slots as u32, self.rope as u32, &self.kpe);
        let mut indptr = vec![0i32];
        let mut indices = Vec::new();
        for l in &self.lanes {
            indices.extend_from_slice(l);
            indptr.push(indices.len() as i32);
        }
        KvPool {
            keys,
            values,
            page_indices: b.i32(indices.len() as u32, 1, &indices),
            page_indptr: b.i32(indptr.len() as u32, 1, &indptr),
            page_size: self.ps as i32,
            max_pages: self.max_pages,
            seq_stride: self.rank as u64,
            head_stride: self.rank as u64,
        }
    }
}

/// The MLA reading of row `r`: softmax over `keys` of `(q·ckv + q_pe·kpe)·scale`,
/// answering the weighted latent.
fn mla_ref(
    pool: &Pool,
    q: &[f32],
    qpe: &[f32],
    heads: usize,
    r: usize,
    keys: &[usize],
    scale: f64,
) -> Vec<f32> {
    let (rk, rp) = (pool.rank, pool.rope);
    let mut out = vec![0.0f32; heads * rk];
    if keys.is_empty() {
        return out;
    }
    for h in 0..heads {
        let s: Vec<f64> = keys
            .iter()
            .map(|&sl| {
                let a: f64 = (0..rk)
                    .map(|i| {
                        f64::from(q[(r * heads + h) * rk + i]) * f64::from(pool.ckv[sl * rk + i])
                    })
                    .sum();
                let b: f64 = (0..rp)
                    .map(|i| {
                        f64::from(qpe[(r * heads + h) * rp + i]) * f64::from(pool.kpe[sl * rp + i])
                    })
                    .sum();
                (a + b) * scale
            })
            .collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = e.iter().sum();
        for i in 0..rk {
            let acc: f64 = keys
                .iter()
                .zip(&e)
                .map(|(&sl, w)| w * f64::from(pool.ckv[sl * rk + i]))
                .sum();
            out[h * rk + i] = round_bf16((acc / l) as f32);
        }
    }
    out
}

#[test]
fn mla_attention_reads_latents_through_pages() {
    let mut rng = Rng(3);
    let (heads, rank, rope, ps) = (4usize, 64usize, 16usize, 8usize);
    let pages = 9usize;
    let slots = pages * ps;
    let pool = Pool {
        ps,
        rank,
        rope,
        lanes: vec![vec![6, 2, 8, 0], vec![1, 5], vec![3, 7, 4]],
        ckv: rng.bf16s(slots * rank, 1.0),
        kpe: rng.bf16s(slots * rope, 1.0),
        slots,
        max_pages: 4,
    };
    // Prefill: 7 rows of lane 0, 4 of lane 1, 1 of lane 2, one padded row.
    let mut rows: Vec<(usize, i32)> = (0..7).map(|i| (0, 24 + i)).collect();
    rows.extend((0..4).map(|i| (1, 9 + i)));
    rows.push((2, 20));
    rows.push((0, -1));
    let n = rows.len();
    let positions: Vec<i32> = rows.iter().map(|r| r.1).collect();
    let req: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
    let q = rng.bf16s(n * heads * rank, 1.0);
    let qpe = rng.bf16s(n * heads * rope, 1.0);
    let top_k = 6usize;
    let sel: Vec<i32> = (0..n * top_k).map(|i| ((i * 11) % 40) as i32 - 3).collect();
    let scale = 0.09f32;
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let pt = b.i32(n as u32, 1, &positions);
    let rt = b.i32(n as u32, 1, &req);
    let qt = b.bf16(n as u32, (heads * rank) as u32, &q);
    let qr = RaggedTensor {
        data: qt,
        indptr: b.i32(4, 1, &[0, 7, 11, 12]),
    };
    let qp = b.bf16(n as u32, (heads * rope) as u32, &qpe);
    let st = b.i32(n as u32, top_k as u32, &sel);
    let w = (heads * rank) as u32;
    let outs: Vec<_> = (0..4).map(|_| b.zeros(Dtype::Bf16, n as u32, w)).collect();
    let (h, r) = (heads as u32, rank as u32);
    if !b
        .run(|ctx| {
            mla::attention_prefill(ctx, qr, qp, &kv, pt, rt, h, r, scale, outs[0])?;
            mla::attention_decode(ctx, qt, qp, &kv, pt, rt, h, r, scale, outs[1])?;
            mla::attention_prefill_selected(ctx, qr, qp, st, &kv, pt, rt, h, r, scale, outs[2])?;
            mla::attention_decode_selected(ctx, qt, qp, st, &kv, pt, rt, h, r, scale, outs[3])
        })
        .unwrap()
    {
        return;
    }
    let dense = |r: usize| -> Vec<usize> {
        let (lane, qpos) = rows[r];
        (0..=qpos).map(|kp| pool.slot(lane, kp as usize)).collect()
    };
    let chosen = |r: usize| -> Vec<usize> {
        let (lane, qpos) = rows[r];
        sel[r * top_k..(r + 1) * top_k]
            .iter()
            .filter(|&&j| j >= 0 && j <= qpos)
            .map(|&j| pool.slot(lane, j as usize))
            .collect()
    };
    let live = n - 1;
    for (i, which) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
        let mut want = Vec::new();
        for r in 0..n {
            let keys = if r >= live {
                Vec::new()
            } else if which == 0 {
                dense(r)
            } else {
                chosen(r)
            };
            want.extend(mla_ref(&pool, &q, &qpe, heads, r, &keys, f64::from(scale)));
        }
        let got = b.read_f32(outs[i]);
        assert!(got.iter().all(|x| x.is_finite()));
        assert_close(
            &got[..live * heads * rank],
            &want[..live * heads * rank],
            1e-3,
            8e-3,
        );
    }
}

#[test]
fn mla_kv_append_writes_both_planes_and_drops_padding() {
    let mut rng = Rng(4);
    let (rank, rope, ps, pages) = (32usize, 8usize, 4usize, 3usize);
    let slots = ps * pages;
    let pool = Pool {
        ps,
        rank,
        rope,
        lanes: vec![vec![0]],
        ckv: rng.bf16s(slots * rank, 1.0),
        kpe: rng.bf16s(slots * rope, 1.0),
        slots,
        max_pages: 1,
    };
    let n = 4usize;
    let c = rng.bf16s(n * rank, 1.0);
    let p = rng.bf16s(n * rope, 1.0);
    let page = [2i32, i32::MAX, 0, 1];
    let off = [3i32, 0, 1, 4];
    let mut b = Bench::new();
    let kv = pool.bind(&mut b);
    let ct = b.bf16(n as u32, rank as u32, &c);
    let ptt = b.bf16(n as u32, rope as u32, &p);
    let wp = b.i32(n as u32, 1, &page);
    let wo = b.i32(n as u32, 1, &off);
    if !b
        .run(|ctx| mla::kv_append(ctx, ct, ptt, &kv, wp, wo))
        .unwrap()
    {
        return;
    }
    let mut want_c = pool.ckv.clone();
    let mut want_p = pool.kpe.clone();
    for i in 0..n {
        if page[i] < 0 || page[i] as usize >= pages || off[i] as usize >= ps {
            continue;
        }
        let s = page[i] as usize * ps + off[i] as usize;
        want_c[s * rank..(s + 1) * rank].copy_from_slice(&c[i * rank..(i + 1) * rank]);
        want_p[s * rope..(s + 1) * rope].copy_from_slice(&p[i * rope..(i + 1) * rope]);
    }
    assert_close(&b.read_f32(kv.keys), &want_c, 0.0, 0.0);
    assert_close(&b.read_f32(kv.values), &want_p, 0.0, 0.0);
}
