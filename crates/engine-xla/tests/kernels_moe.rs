//! The MoE family (routers, routed matmuls, combines) against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::Bank;
use kernels_xla::linear::moe;

fn hash(i: usize, seed: u32) -> u32 {
    (i as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(40503))
        .rotate_left(13)
        .wrapping_mul(0x9E37_79B1)
}

/// bf16-exact values in `[-1, 1)`.
fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| round_bf16(((hash(i, seed) >> 8) % 2000) as f32 / 1000.0 - 1.0))
        .collect()
}

/// Router logits: each row a permutation of distinct steps of 0.25 around 0.
fn logits(rows: usize, width: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        let mut idx: Vec<usize> = (0..width).collect();
        idx.sort_by_key(|&i| hash(r * 1000 + i, seed));
        for &i in &idx {
            out.push(round_bf16((i as f32 - width as f32 / 2.0) * 0.25));
        }
    }
    out
}

/// Indices of the `k` largest ranks, ties to the lowest index.
fn topk(rank: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..rank.len()).collect();
    idx.sort_by(|&a, &b| rank[b].partial_cmp(&rank[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sqrt_softplus(x: f32) -> f32 {
    let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    sp.max(0.0).sqrt()
}

fn renorm(w: &mut [f32], renormalize: bool, scaling: f32) {
    let sum: f32 = w.iter().sum();
    let scale = if renormalize && sum > 0.0 {
        scaling / sum
    } else {
        scaling
    };
    for v in w {
        *v *= scale;
    }
}

#[test]
fn the_routers_answer_the_host() {
    let (t, e, k) = (5usize, 12usize, 3usize);
    let lg = logits(t, e, 1);
    let gain = data(e, 2);
    let corr: Vec<f32> = data(e, 3).iter().map(|v| v * 0.1).collect();
    let (sink, se) = (2usize, 10usize);
    let lg_sink = logits(t, se + sink, 4);
    let corr_sink: Vec<f32> = data(se, 5).iter().map(|v| v * 0.1).collect();

    let mut b = Bench::new();
    let x = b.bf16(t as u32, e as u32, &lg);
    let xs = b.bf16(t as u32, (se + sink) as u32, &lg_sink);
    let g = b.bf16(1, e as u32, &gain);
    let c = b.f32(1, e as u32, &corr);
    let cs = b.f32(1, se as u32, &corr_sink);
    let gs = b.f32(1, 1, &[0.75]);
    let mut outs = Vec::new();
    for _ in 0..6 {
        outs.push((
            b.zeros(Dtype::I32, t as u32, k as u32),
            b.zeros(Dtype::F32, t as u32, k as u32),
        ));
    }
    let wide = (
        b.zeros(Dtype::I32, t as u32, 14),
        b.zeros(Dtype::F32, t as u32, 14),
    );
    let sunk = (
        b.zeros(Dtype::I32, t as u32, (k + sink) as u32),
        b.zeros(Dtype::F32, t as u32, (k + sink) as u32),
    );
    let groups = b.zeros(Dtype::I32, t as u32, 4);
    let o = outs.clone();
    if !b
        .run(|ctx| {
            moe::topk_softmax(ctx, x, e as u32, k as u32, o[0].0, o[0].1)?;
            moe::topk_softmax_scaled(ctx, x, g, e as u32, k as u32, o[1].0, o[1].1)?;
            moe::topk_sigmoid(ctx, x, e as u32, k as u32, true, 2.5, o[2].0, o[2].1)?;
            moe::topk_sigmoid_biased(ctx, x, c, e as u32, k as u32, false, 1.5, o[3].0, o[3].1)?;
            moe::topk_sqrt_softplus(ctx, x, c, e as u32, k as u32, true, 1.0, o[4].0, o[4].1)?;
            moe::predict_route(ctx, x, c, e as u32, k as u32, o[5].0, o[5].1)?;
            // A fan-out past the expert count: the tail lands expert 0 at weight 0.
            moe::topk_sigmoid(ctx, x, e as u32, 14, true, 1.0, wide.0, wide.1)?;
            moe::topk_sigmoid_sink(
                ctx,
                xs,
                Some(cs),
                Some(gs),
                se as u32,
                k as u32,
                sink as u32,
                2.0,
                sunk.0,
                sunk.1,
            )?;
            moe::group_routes(ctx, 4, groups)
        })
        .unwrap()
    {
        return;
    }

    let mut want_r: Vec<Vec<i32>> = vec![Vec::new(); 6];
    let mut want_w: Vec<Vec<f32>> = vec![Vec::new(); 6];
    let (mut wide_r, mut wide_w) = (Vec::new(), Vec::new());
    for r in 0..t {
        let row = &lg[r * e..(r + 1) * e];
        // softmax, plain and scaled
        let pick = topk(row, k);
        let mx = pick.iter().map(|&i| row[i]).fold(f32::MIN, f32::max);
        let sum: f32 = pick.iter().map(|&i| (row[i] - mx).exp()).sum();
        for &i in &pick {
            let w = (row[i] - mx).exp() / sum;
            want_r[0].push(i as i32);
            want_w[0].push(w);
            want_r[1].push(i as i32);
            want_w[1].push(w * gain[i]);
        }
        // sigmoid family
        let sig: Vec<f32> = row.iter().map(|&v| sigmoid(v)).collect();
        let ssp: Vec<f32> = row.iter().map(|&v| sqrt_softplus(v)).collect();
        let cases: [(usize, &[f32], bool, bool, f32); 4] = [
            (2, &sig, false, true, 2.5),
            (3, &sig, true, false, 1.5),
            (4, &ssp, true, true, 1.0),
            (5, &ssp, true, false, 1.0),
        ];
        for (slot, score, biased, renormalize, scaling) in cases {
            let rank: Vec<f32> = (0..e)
                .map(|i| score[i] + if biased { corr[i] } else { 0.0 })
                .collect();
            let pick = topk(&rank, k);
            let mut w: Vec<f32> = pick.iter().map(|&i| score[i]).collect();
            renorm(&mut w, renormalize, scaling);
            want_r[slot].extend(pick.iter().map(|&i| i as i32));
            want_w[slot].extend(w);
        }
        let pick = topk(&sig, e);
        let mut w: Vec<f32> = pick.iter().map(|&i| sig[i]).collect();
        w.extend([0.0, 0.0]);
        renorm(&mut w, true, 1.0);
        wide_r.extend(pick.iter().map(|&i| i as i32));
        wide_r.extend([0, 0]);
        wide_w.extend(w);
    }
    for s in 0..6 {
        assert_eq!(b.read_i32(outs[s].0), want_r[s], "router {s} routes");
        assert_close(&b.read_f32(outs[s].1), &want_w[s], 1e-6, 1e-4);
    }
    assert_eq!(b.read_i32(wide.0), wide_r);
    assert_close(&b.read_f32(wide.1), &wide_w, 1e-6, 1e-4);

    let (mut sr, mut sw) = (Vec::new(), Vec::new());
    for r in 0..t {
        let row = &lg_sink[r * (se + sink)..(r + 1) * (se + sink)];
        let sig: Vec<f32> = row.iter().map(|&v| sigmoid(v)).collect();
        let rank: Vec<f32> = (0..se).map(|i| sig[i] + corr_sink[i]).collect();
        let pick = topk(&rank, k);
        let mut w: Vec<f32> = pick.iter().map(|&i| sig[i]).collect();
        let mut ids: Vec<i32> = pick.iter().map(|&i| i as i32).collect();
        for s in 0..sink {
            ids.push((se + s) as i32);
            w.push(sig[se + s]);
        }
        let sum: f32 = w.iter().sum();
        let scale = 2.0 * 0.75 / (sum + 1e-20);
        sr.extend(ids);
        sw.extend(w.iter().map(|v| v * scale));
    }
    assert_eq!(b.read_i32(sunk.0), sr);
    assert_close(&b.read_f32(sunk.1), &sw, 1e-6, 1e-4);

    let want_g: Vec<i32> = (0..t).flat_map(|_| 0..4).collect();
    assert_eq!(b.read_i32(groups), want_g);
}

#[test]
fn a_hash_route_reads_its_table_and_weighs_the_named_experts() {
    let (t, e, k, vocab) = (6usize, 9usize, 3usize, 7usize);
    let lg = logits(t, e, 7);
    // Token ids: in range, past the vocabulary, and negative (a huge u32).
    let ids: Vec<i32> = vec![3, 0, 6, 9, -1, 5];
    let mut table: Vec<i64> = (0..vocab * k)
        .map(|i| (hash(i, 8) % e as u32) as i64)
        .collect();
    table[5 * k + 1] = -1; // outside the experts: weight 0, route -1
    table[3 * k + 2] = (e + 3) as i64;
    let mut b = Bench::new();
    let id = b.i32(t as u32, 1, &ids);
    let tab = b.i64(vocab as u32, k as u32, &table);
    let x = b.bf16(t as u32, e as u32, &lg);
    let r = b.zeros(Dtype::I32, t as u32, k as u32);
    let w = b.zeros(Dtype::F32, t as u32, k as u32);
    if !b
        .run(|ctx| moe::hash_route(ctx, id, tab, x, vocab as u32, k as u32, true, 1.5, r, w))
        .unwrap()
    {
        return;
    }
    let (mut wr, mut ww) = (Vec::new(), Vec::new());
    for row in 0..t {
        let raw = ids[row] as u32 as usize;
        let tid = if raw < vocab { raw } else { 0 };
        let mut wts = Vec::new();
        for s in 0..k {
            let eid = table[tid * k + s];
            wr.push(eid as i32);
            wts.push(if (0..e as i64).contains(&eid) {
                let v = lg[row * e + eid as usize];
                let sp = if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
                sp.sqrt()
            } else {
                0.0
            });
        }
        renorm(&mut wts, true, 1.5);
        ww.extend(wts);
    }
    assert_eq!(b.read_i32(r), wr);
    assert_close(&b.read_f32(w), &ww, 1e-6, 1e-4);
}

#[test]
fn the_combines_answer_the_host() {
    let (t, k, w, e) = (5usize, 3usize, 40usize, 6usize);
    let routed = data(t * k * w, 11);
    let wts: Vec<f32> = data(t * k, 12);
    let xs = data(t * w, 13);
    let bias = data(e * w, 14);
    let mut routes: Vec<i32> = (0..t * k)
        .map(|i| (hash(i, 15) % e as u32) as i32)
        .collect();
    routes[4] = -1;
    let shared = data(t * w, 16);
    let gate = data(t, 17);

    let mut b = Bench::new();
    let rt = b.bf16((t * k) as u32, w as u32, &routed);
    let wt = b.f32(t as u32, k as u32, &wts);
    let x = b.bf16(t as u32, w as u32, &xs);
    let bi = b.bf16(e as u32, w as u32, &bias);
    let ro = b.i32(t as u32, k as u32, &routes);
    let sh = b.bf16(t as u32, w as u32, &shared);
    let ga = b.bf16(t as u32, 1, &gate);
    let tk = b.bf16(t as u32, w as u32, &xs);
    let y1 = b.zeros(Dtype::Bf16, t as u32, w as u32);
    let y2 = b.zeros(Dtype::Bf16, t as u32, w as u32);
    let y3 = b.zeros(Dtype::Bf16, t as u32, w as u32);
    if !b
        .run(|ctx| {
            moe::weighted_sum(ctx, rt, wt, y1)?;
            moe::bias_sum(ctx, x, bi, ro, wt, y2)?;
            moe::sigmoid_gate_add(ctx, tk, sh, ga, y3)
        })
        .unwrap()
    {
        return;
    }
    let mut w1 = vec![0.0; t * w];
    let mut w2 = vec![0.0; t * w];
    let mut w3 = vec![0.0; t * w];
    for r in 0..t {
        for c in 0..w {
            let mut acc = 0.0;
            let mut bacc = xs[r * w + c];
            for s in 0..k {
                acc += wts[r * k + s] * routed[(r * k + s) * w + c];
                let ex = routes[r * k + s];
                if ex >= 0 {
                    bacc += wts[r * k + s] * bias[ex as usize * w + c];
                }
            }
            w1[r * w + c] = round_bf16(acc);
            w2[r * w + c] = round_bf16(bacc);
            w3[r * w + c] = round_bf16(xs[r * w + c] + sigmoid(gate[r]) * shared[r * w + c]);
        }
    }
    assert_close(&b.read_f32(y1), &w1, 1e-2, 1e-2);
    assert_close(&b.read_f32(y2), &w2, 1e-2, 1e-2);
    assert_close(&b.read_f32(y3), &w3, 1e-2, 1e-2);
}

// ---------------------------------------------------------------- matmuls

/// A host expert bank: dequantized f32 weights `[E, N, K]` plus the planes a
/// kernel reads.
struct HostBank {
    w: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<f32>,
    biases: Option<Vec<f32>>,
    scale_bytes: Vec<u8>,
    group: u32,
    bits: u32,
    /// Land the codes as little-endian u32 words instead of bytes.
    words: bool,
}

fn pack(codes: &[u32], bits: u32) -> Vec<u8> {
    let per = (8 / bits) as usize;
    codes
        .chunks(per)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |acc, (j, &q)| acc | ((q as u8) << (j as u32 * bits)))
        })
        .collect()
}

fn affine_bank(e: usize, n: usize, k: usize, group: u32, bits: u32, seed: u32) -> HostBank {
    let g = group as usize;
    let q: Vec<u32> = (0..e * n * k)
        .map(|i| hash(i, seed) % (1 << bits))
        .collect();
    let scales: Vec<f32> = (0..e * n * k / g)
        .map(|i| round_bf16(0.01 + (hash(i, seed + 1) % 50) as f32 / 1000.0))
        .collect();
    let biases: Vec<f32> = (0..e * n * k / g)
        .map(|i| round_bf16((hash(i, seed + 2) % 400) as f32 / 1000.0 - 0.2))
        .collect();
    let w = (0..e * n * k)
        .map(|i| scales[i / g] * q[i] as f32 + biases[i / g])
        .collect();
    HostBank {
        w,
        codes: pack(&q, bits),
        scales,
        biases: Some(biases),
        scale_bytes: Vec::new(),
        group,
        bits,
        words: false,
    }
}

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn mxfp4_bank(e: usize, n: usize, k: usize, seed: u32) -> HostBank {
    let q: Vec<u32> = (0..e * n * k).map(|i| hash(i, seed) % 16).collect();
    let sb: Vec<u8> = (0..e * n * k / 32)
        .map(|i| 122 + (hash(i, seed + 1) % 5) as u8)
        .collect();
    let w = (0..e * n * k)
        .map(|i| E2M1[q[i] as usize] * 2f32.powi(i32::from(sb[i / 32]) - 127))
        .collect();
    HostBank {
        w,
        codes: pack(&q, 4),
        scales: Vec::new(),
        biases: None,
        scale_bytes: sb,
        group: 32,
        bits: 4,
        words: false,
    }
}

fn dense_bank(e: usize, n: usize, k: usize, seed: u32) -> HostBank {
    let w: Vec<f32> = data(e * n * k, seed)
        .iter()
        .map(|v| round_bf16(v * 0.25))
        .collect();
    HostBank {
        w,
        codes: Vec::new(),
        scales: Vec::new(),
        biases: None,
        scale_bytes: Vec::new(),
        group: 0,
        bits: 16,
        words: false,
    }
}

impl HostBank {
    fn land(&self, b: &mut Bench, e: usize) -> Bank {
        let e32 = e as u32;
        let codes = if self.words {
            let w: Vec<u32> = self
                .codes
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            b.u32(e32, (w.len() / e) as u32, &w)
        } else {
            b.u8(e32, (self.codes.len() / e) as u32, &self.codes)
        };
        let (scales, biases) = match &self.biases {
            Some(bi) => (
                b.bf16(e32, (self.scales.len() / e) as u32, &self.scales),
                Some(b.bf16(e32, (bi.len() / e) as u32, bi)),
            ),
            None => (
                b.u8(e32, (self.scale_bytes.len() / e) as u32, &self.scale_bytes),
                None,
            ),
        };
        Bank {
            codes,
            scales,
            biases,
            group: self.group,
            bits: self.bits,
        }
    }

    fn dense(&self, b: &mut Bench, e: usize) -> kernels_xla::Tensor {
        b.bf16(e as u32, (self.w.len() / e) as u32, &self.w)
    }
}

/// `y[p] = W[routes[p]] · x_row(p) (+ bias)`, rows with a route outside the
/// bank keep `prev`.
#[allow(clippy::too_many_arguments)]
fn select_ref(
    w: &[f32],
    e: usize,
    n: usize,
    k: usize,
    x: &[f32],
    per_token: bool,
    routes: &[i32],
    top_k: usize,
    bias: Option<&[f32]>,
    prev: f32,
) -> Vec<f32> {
    let mut y = vec![prev; routes.len() * n];
    for (p, &r) in routes.iter().enumerate() {
        if r < 0 || r as usize >= e {
            continue;
        }
        let r = r as usize;
        let xr = if per_token { p / top_k } else { p };
        for c in 0..n {
            let mut acc: f32 = (0..k).map(|j| w[(r * n + c) * k + j] * x[xr * k + j]).sum();
            if let Some(bi) = bias {
                acc += bi[r * n + c];
            }
            y[p * n + c] = round_bf16(acc);
        }
    }
    y
}

const PREV: f32 = 7.0;

/// Runs one select entry over a small (gather) and a large (ragged) fire.
fn check_select(bank: &HostBank, e: usize, n: usize, k: usize, which: &str) {
    let bias = data(e * n, 21);
    // (tokens, top_k): 2x2 = 4 pairs < e gathers; 5x3 = 15 pairs sorts;
    // then 15 pairs crowding one expert (more rows than a tile of the loop).
    for (t, tk, crowded) in [(2usize, 2usize, false), (5, 3, false), (5, 3, true)] {
        let x = data(t * k, 22);
        let mut routes: Vec<i32> = (0..t * tk)
            .map(|i| (hash(i, 23 + t as u32) % e as u32) as i32)
            .collect();
        if crowded {
            for (i, r) in routes.iter_mut().enumerate() {
                if i % 7 != 3 {
                    *r = 1;
                }
            }
        }
        routes[1] = -1;
        let mut b = Bench::new();
        let xt = b.bf16(t as u32, k as u32, &x);
        let rt = b.i32(t as u32, tk as u32, &routes);
        let bt = b.bf16(e as u32, n as u32, &bias);
        let prev = vec![PREV; t * tk * n];
        let y = b.bf16((t * tk) as u32, n as u32, &prev);
        let biased = which == "bias";
        let run = match which {
            "dense" => {
                let d = bank.dense(&mut b, e);
                b.run(|ctx| moe::matmul_select(ctx, xt, d, rt, y))
            }
            _ => {
                let bk = bank.land(&mut b, e);
                if biased {
                    b.run(|ctx| moe::matmul_select_bias(ctx, xt, bk, bt, rt, y))
                } else {
                    b.run(|ctx| moe::matmul_select_quant(ctx, xt, bk, rt, y))
                }
            }
        };
        if !run.unwrap_or_else(|err| panic!("{which}: {err}\n{}", b.last_module)) {
            return;
        }
        let want = select_ref(
            &bank.w,
            e,
            n,
            k,
            &x,
            true,
            &routes,
            tk,
            biased.then_some(&bias[..]),
            PREV,
        );
        let got = b.read_f32(y);
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_close(&got, &want, 0.01 * scale.max(1.0), 1e-2);
    }
}

#[test]
fn a_dense_select_answers_the_host_gathered_and_sorted() {
    let (e, n, k) = (6, 24, 96);
    check_select(&dense_bank(e, n, k, 30), e, n, k, "dense");
}

#[test]
fn affine_selects_answer_the_host_at_every_routed_point() {
    let (e, n, k) = (6, 24, 384);
    for (group, bits) in [(64, 4), (64, 2), (32, 2), (128, 2), (64, 8)] {
        check_select(
            &affine_bank(e, n, k, group, bits, 40 + bits),
            e,
            n,
            k,
            "quant",
        );
    }
    // The same bank landed as u32 words.
    let mut words = affine_bank(e, n, k, 64, 4, 70);
    words.words = true;
    check_select(&words, e, n, k, "quant");
}

#[test]
fn mxfp4_selects_answer_the_host_with_and_without_a_bias() {
    let (e, n, k) = (6, 24, 160);
    let bank = mxfp4_bank(e, n, k, 50);
    check_select(&bank, e, n, k, "quant");
    check_select(&bank, e, n, k, "bias");
    let mut words = mxfp4_bank(e, n, k, 51);
    words.words = true;
    check_select(&words, e, n, k, "quant");
}

#[test]
fn a_grouped_matmul_runs_each_slice_through_its_expert() {
    let (rows, groups, k, n) = (3usize, 4usize, 64usize, 16usize);
    let dense = dense_bank(groups, n, k, 60);
    let quant = affine_bank(groups, n, k, 64, 4, 61);
    let x = data(rows * groups * k, 62);
    let mut b = Bench::new();
    let xt = b.bf16(rows as u32, (groups * k) as u32, &x);
    let rt = b.zeros(Dtype::I32, rows as u32, groups as u32);
    let d = dense.dense(&mut b, groups);
    let q = quant.land(&mut b, groups);
    let y1 = b.zeros(Dtype::Bf16, rows as u32, (groups * n) as u32);
    let y2 = b.zeros(Dtype::Bf16, rows as u32, (groups * n) as u32);
    if !b
        .run(|ctx| {
            moe::group_routes(ctx, groups as u32, rt)?;
            moe::matmul_grouped(ctx, xt, moe::GroupedPlane::Dense(d), rt, groups as u32, y1)?;
            moe::matmul_grouped(ctx, xt, moe::GroupedPlane::Bank(q), rt, groups as u32, y2)
        })
        .unwrap_or_else(|err| panic!("{err}\n{}", b.last_module))
    {
        return;
    }
    let routes: Vec<i32> = (0..rows).flat_map(|_| 0..groups as i32).collect();
    for (bank, y) in [(&dense, y1), (&quant, y2)] {
        let want = select_ref(&bank.w, groups, n, k, &x, false, &routes, groups, None, 0.0);
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert_close(&b.read_f32(y), &want, 0.01 * scale.max(1.0), 1e-2);
    }
}

/// An e5m2-exact weight (what `kernels_xla::pack` lands pre-scaled mxfp4 as).
fn e5m2_data(n: usize, seed: u32) -> Vec<f32> {
    const V: [f32; 8] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
    (0..n)
        .map(|i| {
            let h = hash(i, seed);
            let v = V[(h % 8) as usize];
            if h & 256 != 0 { -v } else { v }
        })
        .collect()
}

/// The grouped-matmul kernel (`kernels_xla::mosaic::gmm`) at the three
/// cuts it makes of a bank: the TPU's `E·N`-minor layout in whole
/// 128-lane blocks and in per-expert element windows (N not whole lane
/// tiles: gpt-oss's down bank), and a row-major bank; e5m2 and bf16
/// weights, per-token and per-pair rows, experts with more rows than a
/// tile, unrouted pairs, a bias. `PIE_XLA_MOE_GMM=0` or another forced
/// path checks the same cases on the XLA paths.
#[test]
fn the_grouped_matmul_kernel_answers_the_host() {
    let forced = std::env::var("PIE_XLA_MOE_PATH").ok();
    // (experts, N, K, weights e5m2): K 160 is not whole lane tiles, so the
    // TPU lays the [E·N, K] bank E·N-minor.
    for (e, n, k, f8) in [
        (4usize, 256usize, 160usize, true),
        (4, 192, 160, true),
        (5, 64, 256, false),
    ] {
        let w = if f8 {
            e5m2_data(e * n * k, 90 + n as u32)
        } else {
            data(e * n * k, 91)
                .iter()
                .map(|v| round_bf16(v * 0.25))
                .collect()
        };
        let bias = data(e * n, 92);
        for (t, tk, per_token, crowded) in [
            (9usize, 3usize, true, false),
            (20, 4, true, true),
            (12, 2, false, false),
        ] {
            let rows = if per_token { t } else { t * tk };
            let x = data(rows * k, 93 + t as u32);
            let mut routes: Vec<i32> = (0..t * tk)
                .map(|i| (hash(i, 94 + t as u32) % e as u32) as i32)
                .collect();
            if crowded {
                // Most of the pairs on expert 2 (two tiles and a bit), some
                // repeats within a token.
                for (i, r) in routes.iter_mut().enumerate() {
                    if i % 5 != 0 {
                        *r = 2;
                    }
                }
            }
            routes[1] = -1;
            routes[t * tk - 1] = e as i32;
            let mut b = Bench::new();
            let xt = b.bf16(rows as u32, k as u32, &x);
            let rt = b.i32(t as u32, tk as u32, &routes);
            let bt = b.bf16(e as u32, n as u32, &bias);
            let y = b.bf16((t * tk) as u32, n as u32, &vec![PREV; t * tk * n]);
            let biased = f8 && !per_token;
            let run = if f8 {
                let bytes: Vec<u8> = w
                    .iter()
                    .map(|&v| (kernels_xla::hlo::f16_bits(v) >> 8) as u8)
                    .collect();
                let codes = b.raw(Dtype::E5m2, (e * n) as u32, k as u32, bytes);
                let scales = b.u8((e * n) as u32, (k / 32) as u32, &vec![127; e * n * k / 32]);
                let bank = Bank {
                    codes,
                    scales,
                    biases: None,
                    group: 32,
                    bits: 4,
                };
                if biased {
                    b.run(|ctx| moe::matmul_select_bias(ctx, xt, bank, bt, rt, y))
                } else {
                    b.run(|ctx| moe::matmul_select_quant(ctx, xt, bank, rt, y))
                }
            } else {
                let d = b.bf16((e * n) as u32, k as u32, &w);
                b.run(|ctx| moe::matmul_select(ctx, xt, d, rt, y))
            };
            if !run.unwrap_or_else(|err| panic!("{err}\n{}", b.last_module)) {
                return;
            }
            let off = std::env::var("PIE_XLA_MOE_GMM").is_ok_and(|v| v == "0");
            if forced.as_deref() == Some("gmm") || (forced.is_none() && !off) {
                assert!(
                    b.last_module.contains("tpu_custom_call"),
                    "the kernel ran for {e}x{n}x{k}"
                );
            }
            let want = select_ref(
                &w,
                e,
                n,
                k,
                &x,
                per_token,
                &routes,
                tk,
                biased.then_some(&bias[..]),
                PREV,
            );
            let got = b.read_f32(y);
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert_close(&got, &want, 0.01 * scale.max(1.0), 1e-2);
        }
    }
}
