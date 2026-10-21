//! The sparse indexer and the compressed KV pools against host references:
//! the indexer's rotations and key norm, its exact top-k over paged
//! (optionally compressed) keys, the pools' boundaries, state, gather,
//! block mean and store, and the attention over a pool, dense and selected,
//! on the latent walk (512-wide entries) and the reference walk.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, KV, bf16s, close_abs, close_bf16, round_bf16};
use dtype::Dtype;
use kernels_rocm::attn::{index, pool};
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

/// Page tables for `cells[r]` cells of request r at `ps` a page, pages in
/// reverse: (indices, indptr, last page lengths, total pages).
fn pages(cells: &[usize], ps: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, usize) {
    let per: Vec<usize> = cells.iter().map(|&n| n.div_ceil(ps).max(1)).collect();
    let total = per.iter().sum::<usize>() + 1;
    let mut indices = Vec::new();
    let mut indptr = vec![0i32];
    let mut next = total as i32 - 1;
    for &p in &per {
        for _ in 0..p {
            indices.push(next);
            next -= 1;
        }
        indptr.push(indices.len() as i32);
    }
    let last = cells
        .iter()
        .zip(&per)
        .map(|(&n, &p)| (n - (p - 1) * ps) as i32)
        .collect();
    (indices, indptr, last, total)
}

/// A one-plane pool of `width`-wide rows: request r's cell i holds
/// `rows[r][i]` (all of them `cells[r]` long).
struct Pool {
    pool: KvPool,
    indices: Vec<i32>,
    indptr: Vec<i32>,
    ps: usize,
}

impl Pool {
    fn cell(&self, r: usize, pos: usize) -> usize {
        self.indices[self.indptr[r] as usize + pos / self.ps] as usize * self.ps + pos % self.ps
    }
}

fn make_pool(gpu: &Gpu, rows: &[Vec<f32>], cells: &[usize], width: usize, ps: usize) -> Pool {
    let (indices, indptr, last, total) = pages(cells, ps);
    let mut plane = vec![0f32; total * ps * width];
    for (r, &n) in cells.iter().enumerate() {
        for i in 0..n {
            let cell = indices[indptr[r] as usize + i / ps] as usize * ps + i % ps;
            plane[cell * width..(cell + 1) * width]
                .copy_from_slice(&rows[r][i * width..(i + 1) * width]);
        }
    }
    let max_pages = indptr.windows(2).map(|w| w[1] - w[0]).max().unwrap();
    let pool = KvPool {
        keys: gpu.tensor(&plane, (total * ps) as u32, width as u32, Dtype::Bf16),
        page_indices: gpu.column(&indices),
        page_indptr: gpu.column(&indptr),
        last_page_lens: gpu.column(&last),
        page_size: ps as i32,
        seq_stride: width as i64,
        head_stride: width as i64,
        max_pages_per_request: max_pages,
        pages_in_batch: indices.len() as i32,
        live: [0, 0],
        ..KV
    };
    Pool {
        pool,
        indices,
        indptr,
        ps,
    }
}

fn indptr_of(lens: &[usize]) -> Vec<i32> {
    let mut v = vec![0i32];
    for &l in lens {
        v.push(v.last().unwrap() + l as i32);
    }
    v
}

// ------------------------------------------------------------- indexer --

fn rotate_interleaved(v: &mut [f64], rope: usize, pos: i32, theta: f64) {
    for i in 0..rope / 2 {
        let freq = theta.powf(-2.0 * i as f64 / rope as f64);
        let (s, c) = (f64::from(pos) * freq).sin_cos();
        let (a, b) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = a * c - b * s;
        v[2 * i + 1] = b * c + a * s;
    }
}

#[test]
fn index_rotations_and_key_norm() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (rows, heads, d, rope, theta) = (5usize, 3usize, 128usize, 64usize, 10000.0f32);
    let pos: Vec<i32> = vec![0, 7, 300, 4095, 9];

    // q: every head's first 64 turn; rows 1..4 under a window.
    let qv = bf16s(rows * heads * d, 1, 1.0);
    let mut q = gpu.tensor(&qv, rows as u32, (heads * d) as u32, Dtype::Bf16);
    let positions = gpu.column(&pos);
    let win = gpu.ints(&[3, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    index::rope(
        &ctx,
        &mut q,
        positions,
        heads as u32,
        d as u32,
        rope as u32,
        theta,
    )
    .unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&q);
    let mut want = qv.clone();
    for (n, &p) in pos.iter().enumerate().take(4).skip(1) {
        for h in 0..heads {
            let at = (n * heads + h) * d;
            let mut v: Vec<f64> = qv[at..at + d].iter().map(|&x| f64::from(x)).collect();
            rotate_interleaved(&mut v, rope, p, f64::from(theta));
            for j in 0..d {
                want[at + j] = v[j] as f32;
            }
        }
    }
    // A rotation cancels: the error is set by the inputs' scale (1.0).
    close_abs("index q rope", &got, &want, 8e-3);

    // k: LayerNorm then the rotation, of the rounded normed values.
    let kv = bf16s(rows * d, 2, 2.0);
    let w = bf16s(d, 3, 1.0);
    let b = bf16s(d, 4, 0.5);
    let mut k = gpu.tensor(&kv, rows as u32, d as u32, Dtype::Bf16);
    index::layernorm_rope(
        &ctx,
        &mut k,
        gpu.column(&pos),
        gpu.tensor(&w, 1, d as u32, Dtype::Bf16),
        gpu.tensor(&b, 1, d as u32, Dtype::Bf16),
        1e-6,
        rope as u32,
        theta,
    )
    .unwrap();
    let got = gpu.read(&k);
    for n in 0..rows {
        let x: Vec<f64> = kv[n * d..(n + 1) * d]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let mean = x.iter().sum::<f64>() / d as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        let mut y: Vec<f64> = (0..d)
            .map(|j| {
                f64::from(round_bf16(
                    ((x[j] - mean) * inv * f64::from(w[j]) + f64::from(b[j])) as f32,
                ))
            })
            .collect();
        rotate_interleaved(&mut y, rope, pos[n], f64::from(theta));
        let want: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        close_abs("index knorm rope", &got[n * d..(n + 1) * d], &want, 3e-2);
    }
}

/// Host scores of row t's visible keys: Σ_h w_h relu(q_h · k_j).
fn host_scores(
    q: &[f32],
    w: Option<&[f32]>,
    keys: &[f32],
    heads: usize,
    d: usize,
    n: usize,
) -> Vec<f64> {
    (0..n)
        .map(|j| {
            (0..heads)
                .map(|h| {
                    let dot: f64 = (0..d)
                        .map(|e| f64::from(q[h * d + e]) * f64::from(keys[j * d + e]))
                        .sum();
                    dot.max(0.0) * w.map_or(1.0, |w| f64::from(w[h]))
                })
                .sum()
        })
        .collect()
}

/// The ranking over requests of `qo_lens` new rows after `prefix` cached
/// tokens, keys pooled at `ratio`: each row's selection is its exact top-k
/// in ascending key order (up to scores within `tol` of the cut).
#[allow(clippy::too_many_arguments)]
fn run_topk(
    heads: usize,
    d: usize,
    ratio: usize,
    top_k: usize,
    qo_lens: &[usize],
    prefix: &[usize],
    weighted: bool,
) {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let ps = 16;
    let kv_lens: Vec<usize> = qo_lens.iter().zip(prefix).map(|(q, p)| q + p).collect();
    // Key j of request r lives at cell (j + 1)·ratio − 1.
    let mut rows = Vec::new();
    let mut keys = Vec::new();
    for (r, &n) in kv_lens.iter().enumerate() {
        let nk = n / ratio;
        let k = bf16s(nk * d, 10 + r as u64, 1.0);
        let mut cells = vec![0f32; n * d];
        for j in 0..nk {
            let cell = (j + 1) * ratio - 1;
            cells[cell * d..(cell + 1) * d].copy_from_slice(&k[j * d..(j + 1) * d]);
        }
        rows.push(cells);
        keys.push(k);
    }
    let p = make_pool(&gpu, &rows, &kv_lens, d, ps);
    let total: usize = qo_lens.iter().sum();
    let qv = bf16s(total * heads * d, 3, 1.0);
    let wv = bf16s(total * heads, 4, 1.0);
    let q = RaggedTensor {
        data: gpu.tensor(&qv, total as u32, (heads * d) as u32, Dtype::Bf16),
        indptr: gpu.column(&indptr_of(qo_lens)),
    };
    let weights = weighted.then(|| gpu.tensor(&wv, total as u32, heads as u32, Dtype::Bf16));
    let mut sel = gpu.zeros(total as u32, top_k as u32, Dtype::I32);
    index::topk(
        &ctx,
        q,
        weights,
        &p.pool,
        heads as u32,
        d as u32,
        top_k as u32,
        ratio as u32,
        &mut sel,
    )
    .unwrap();
    let got = gpu.read_as::<i32>(&sel);
    let mut t = 0;
    for (r, &ql) in qo_lens.iter().enumerate() {
        for i in 0..ql {
            let abs = kv_lens[r] - ql + i;
            let nkeys = (abs + 1) / ratio;
            let row = &got[t * top_k..(t + 1) * top_k];
            let tag = format!("topk h{heads} d{d} ratio{ratio} req {r} row {i} ({nkeys} keys)");
            if nkeys <= top_k {
                let want: Vec<i32> = (0..top_k as i32)
                    .map(|n| if (n as usize) < nkeys { n } else { -1 })
                    .collect();
                assert_eq!(row, &want[..], "{tag}");
            } else {
                let s = host_scores(
                    &qv[t * heads * d..(t + 1) * heads * d],
                    weighted.then(|| &wv[t * heads..(t + 1) * heads]),
                    &keys[r],
                    heads,
                    d,
                    nkeys,
                );
                assert!(row.windows(2).all(|w| w[0] < w[1]), "{tag}: not ascending");
                assert!(
                    row.iter().all(|&j| j >= 0 && (j as usize) < nkeys),
                    "{tag}: out of range"
                );
                let mut picked = vec![false; nkeys];
                for &j in row {
                    picked[j as usize] = true;
                }
                let lo = row.iter().map(|&j| s[j as usize]).fold(f64::MAX, f64::min);
                let hi = (0..nkeys)
                    .filter(|&j| !picked[j])
                    .map(|j| s[j])
                    .fold(f64::MIN, f64::max);
                let scale = s.iter().fold(0f64, |m, v| m.max(v.abs()));
                assert!(lo >= hi - 2e-3 * scale, "{tag}: picked {lo} < left {hi}");
            }
            t += 1;
        }
    }
}

#[test]
fn topk_ranks_exactly_over_paged_keys() {
    // GLM-5.3's indexer: 32 heads of 128, weighted; decode and prefill rows,
    // short rows listing everything they see.
    run_topk(32, 128, 1, 64, &[1, 1, 5], &[40, 900, 3000], true);
    // A prefill past the candidate lists: rows a block over shared keys.
    run_topk(32, 128, 1, 256, &[260, 300], &[40, 700], true);
    // Past a block's four key lanes: each walks its later key blocks.
    run_topk(32, 128, 1, 128, &[520], &[1700], true);
}

#[test]
fn topk_over_a_compressed_key_pool() {
    // DeepSeek-V4 / GLM-5.3-Flash: keys pooled every 4 tokens.
    run_topk(32, 128, 4, 96, &[1, 3], &[2000, 777], true);
    run_topk(16, 64, 4, 8, &[2], &[150], false);
    run_topk(64, 128, 4, 64, &[530], &[400], true);
}

#[test]
fn topk_takes_any_head_shape() {
    // 80 heads of 96: the thread-a-key scoring.
    run_topk(80, 96, 1, 16, &[1, 2], &[300, 70], true);
}

#[test]
fn topk_breaks_ties_by_key_order() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // Every key alike: the first k are taken, from the candidate list (500
    // ties) and from the walk over the row (3000, more than the list sorts).
    for n in [500usize, 3000] {
        let (heads, d, top_k) = (16usize, 64usize, 37usize);
        let rows = vec![vec![0.25f32; n * d]];
        let p = make_pool(&gpu, &rows, &[n], d, 16);
        let q = RaggedTensor {
            data: gpu.tensor(&vec![0.5f32; heads * d], 1, (heads * d) as u32, Dtype::Bf16),
            indptr: gpu.column(&[0, 1]),
        };
        let mut sel = gpu.zeros(1, top_k as u32, Dtype::I32);
        index::topk(
            &ctx,
            q,
            None,
            &p.pool,
            heads as u32,
            d as u32,
            top_k as u32,
            1,
            &mut sel,
        )
        .unwrap();
        let got = gpu.read_as::<i32>(&sel);
        assert_eq!(got, (0..top_k as i32).collect::<Vec<_>>(), "{n} ties");
    }
}

#[test]
fn index_keys_append_and_block_mean() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (d, ps, ratio) = (128usize, 8usize, 4usize);
    // Request 0: 6 cached + 5 new; request 1: 0 + 3.
    let cells = [11usize, 3];
    let empty = vec![vec![0f32; 11 * d], vec![0f32; 3 * d]];
    let p = make_pool(&gpu, &empty, &cells, d, ps);
    let kv = bf16s(8 * d, 5, 1.0);
    let k = RaggedTensor {
        data: gpu.tensor(&kv, 8, d as u32, Dtype::Bf16),
        indptr: gpu.column(&[0, 5, 8]),
    };
    index::kv_append(&ctx, k, &p.pool, Tensor::ABSENT, Tensor::ABSENT).unwrap();
    let plane = gpu.read(&p.pool.keys);
    let land = [
        (0usize, 6usize),
        (0, 7),
        (0, 8),
        (0, 9),
        (0, 10),
        (1, 0),
        (1, 1),
        (1, 2),
    ];
    for (t, &(r, pos)) in land.iter().enumerate() {
        let c = p.cell(r, pos);
        assert_eq!(
            &plane[c * d..(c + 1) * d],
            &kv[t * d..(t + 1) * d],
            "append row {t}"
        );
    }

    // The mean at boundaries 7 (request 0) and 2 (request 1, cells before
    // 0 count as zero); a non-boundary row gets zeros.
    let bpos = gpu.column(&[7, -1, 2]);
    let breq = gpu.column(&[0, 0, 1]);
    let mut out = gpu.zeros(3, d as u32, Dtype::Bf16);
    index::block_mean(&ctx, bpos, breq, &p.pool, d as u32, ratio as u32, &mut out).unwrap();
    let got = gpu.read(&out);
    let at = |r: usize, pos: usize| {
        let c = p.cell(r, pos);
        plane[c * d..(c + 1) * d].to_vec()
    };
    for j in 0..d {
        let m0: f32 = (4..8).map(|pos| at(0, pos)[j]).sum::<f32>() / 4.0;
        let m2: f32 = (0..3).map(|pos| at(1, pos)[j]).sum::<f32>() / 4.0;
        close_bf16(
            "mean",
            &[got[j], got[d + j], got[2 * d + j]],
            &[m0, 0.0, m2],
            1.0,
        );
    }
}

// --------------------------------------------------------------- pools --

#[test]
fn pool_boundaries_decode_and_prefill() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let pos = [3, 4, 7, 127, 0, 11];
    let n = pos.len() as u32;
    let (mut bp, mut br, mut bro) = (
        gpu.zeros(n, 1, Dtype::I32),
        gpu.zeros(n, 1, Dtype::I32),
        gpu.zeros(n, 1, Dtype::I32),
    );
    pool::boundary_decode(
        &ctx,
        gpu.column(&pos),
        Tensor::ABSENT,
        4,
        &mut bp,
        &mut br,
        &mut bro,
    )
    .unwrap();
    assert_eq!(gpu.read_as::<i32>(&bp), [3, -1, 7, 127, -1, 11]);
    assert_eq!(gpu.read_as::<i32>(&br), [0, 1, 2, 3, 4, 5]);
    assert_eq!(gpu.read_as::<i32>(&bro), [0, 0, 4, 124, 0, 8]);

    let positions = RaggedTensor {
        data: gpu.column(&pos),
        indptr: gpu.column(&[0, 2, 2, 6]),
    };
    pool::boundary_prefill(
        &ctx,
        positions,
        Tensor::ABSENT,
        4,
        &mut bp,
        &mut br,
        &mut bro,
    )
    .unwrap();
    assert_eq!(gpu.read_as::<i32>(&bp), [3, -1, 7, 127, -1, 11]);
    assert_eq!(gpu.read_as::<i32>(&br), [0, 0, 2, 2, 2, 2]);
}

#[test]
fn pool_state_gather_and_store() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (d, ratio, ps) = (64usize, 4usize, 8usize);
    for coff in [1usize, 2] {
        let width = coff * d;
        // A request whose state holds 13 token rows (written one row each).
        let n = 13usize;
        let kv = bf16s(n * width, 20 + coff as u64, 1.0);
        let sc = bf16s(n * width, 30 + coff as u64, 2.0);
        let zero = vec![vec![0f32; n * width]];
        let st_kv = make_pool(&gpu, &zero, &[n], width, ps);
        let st_sc = make_pool(&gpu, &zero, &[n], width, ps);
        let w_page: Vec<i32> = (0..n).map(|i| st_kv.indices[i / ps]).collect();
        let w_off: Vec<i32> = (0..n).map(|i| (i % ps) as i32).collect();
        pool::state_write(
            &ctx,
            gpu.tensor(&kv, n as u32, width as u32, Dtype::Bf16),
            gpu.tensor(&sc, n as u32, width as u32, Dtype::Bf16),
            &st_kv.pool,
            gpu.column(&w_page),
            gpu.column(&w_off),
            d as u32,
            ratio as u32,
            st_kv.pool.keys,
            st_sc.pool.keys,
        )
        .unwrap();
        let ape = bf16s(ratio * width, 40, 0.5);
        // Boundaries at 3, 11 and a non-boundary row.
        let bpos = [3i32, 11, -1];
        let mut entries = gpu.zeros(3, d as u32, Dtype::Bf16);
        pool::gather(
            &ctx,
            gpu.column(&bpos),
            gpu.column(&[0, 0, 0]),
            &st_kv.pool,
            d as u32,
            ratio as u32,
            st_kv.pool.keys,
            st_sc.pool.keys,
            gpu.tensor(&ape, ratio as u32, width as u32, Dtype::F32),
            &mut entries,
        )
        .unwrap();
        let got = gpu.read(&entries);
        let window = coff * ratio;
        for (c, &b) in bpos.iter().enumerate() {
            for j in 0..d {
                let want = if b < 0 {
                    0.0
                } else {
                    let mut terms = Vec::new();
                    for i in 0..window {
                        let pos = b + i as i32 - (window as i32 - 1);
                        if pos < 0 {
                            continue;
                        }
                        let col = if i >= ratio { d } else { 0 } + j;
                        let at = pos as usize * width + col;
                        let s = f64::from(sc[at])
                            + f64::from(ape[(pos as usize % ratio) * width + col]);
                        terms.push((s, f64::from(kv[at])));
                    }
                    let m = terms.iter().map(|t| t.0).fold(f64::MIN, f64::max);
                    let z: f64 = terms.iter().map(|t| (t.0 - m).exp()).sum();
                    (terms.iter().map(|t| (t.0 - m).exp() * t.1).sum::<f64>() / z) as f32
                };
                close_abs(
                    &format!("gather coff{coff} row {c} col {j}"),
                    &[got[c * d + j]],
                    &[want],
                    1e-2,
                );
            }
        }

        // The entries into the pool at their boundary cells.
        let comp = make_pool(&gpu, &[vec![0f32; 16 * d]], &[16], d, ps);
        pool::kv_append(
            &ctx,
            entries,
            gpu.column(&bpos),
            gpu.column(&[0, 0, 0]),
            &comp.pool,
            Tensor::ABSENT,
            Tensor::ABSENT,
        )
        .unwrap();
        let plane = gpu.read(&comp.pool.keys);
        for (c, &b) in bpos.iter().enumerate().take(2) {
            let cell = comp.cell(0, b as usize);
            assert_eq!(&plane[cell * d..(cell + 1) * d], &got[c * d..(c + 1) * d]);
        }
    }
}

/// Attention over a pool of `d`-wide entries (one every `ratio` tokens) for
/// tokens at `positions` of requests `req`, all entries they see or the
/// ones `top_k`-wide selections list.
fn run_pool_lse(
    heads: usize,
    d: usize,
    ratio: usize,
    positions: &[i32],
    req: &[i32],
    top_k: Option<usize>,
) {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let ps = 32;
    let reqs = *req.iter().max().unwrap() as usize + 1;
    let mut cells = vec![1usize; reqs];
    for (&p, &r) in positions.iter().zip(req) {
        cells[r as usize] = cells[r as usize].max(p as usize + 1);
    }
    let mut rows = Vec::new();
    let mut ents = Vec::new();
    for (r, &n) in cells.iter().enumerate() {
        let ne = n / ratio;
        let e = bf16s(ne * d, 50 + r as u64, 1.0);
        let mut plane = vec![0f32; n * d];
        for c in 0..ne {
            let cell = (c + 1) * ratio - 1;
            plane[cell * d..(cell + 1) * d].copy_from_slice(&e[c * d..(c + 1) * d]);
        }
        rows.push(plane);
        ents.push(e);
    }
    let p = make_pool(&gpu, &rows, &cells, d, ps);
    let t = positions.len();
    let qv = bf16s(t * heads * d, 7, 1.0);
    let scale = 1.0 / (d as f32).sqrt();
    let mut sel = Vec::new();
    let lists: Vec<Vec<usize>> = (0..t)
        .map(|i| {
            let visible = (positions[i] as usize + 1) / ratio;
            match top_k {
                None => (0..visible).collect(),
                Some(k) => {
                    let row = topk_row(visible, k, i as u64);
                    sel.extend_from_slice(&row);
                    row.iter()
                        .filter(|&&c| c >= 0)
                        .map(|&c| c as usize)
                        .collect()
                }
            }
        })
        .collect();
    let q = gpu.tensor(&qv, t as u32, (heads * d) as u32, Dtype::Bf16);
    let mut o = gpu.zeros(t as u32, (heads * d) as u32, Dtype::Bf16);
    let mut lse = gpu.zeros(t as u32, heads as u32, Dtype::F32);
    match top_k {
        None => pool::attention_lse(
            &ctx,
            q,
            gpu.column(positions),
            gpu.column(req),
            &p.pool,
            ratio as u32,
            heads as u32,
            d as u32,
            scale,
            &mut o,
            &mut lse,
        ),
        Some(k) => pool::attention_lse_selected(
            &ctx,
            q,
            gpu.column(positions),
            gpu.column(req),
            gpu.ints_shaped(&sel, t as u32, k as u32),
            &p.pool,
            ratio as u32,
            k as u32,
            heads as u32,
            d as u32,
            scale,
            &mut o,
            &mut lse,
        ),
    }
    .unwrap();
    let (got, got_lse) = (gpu.read(&o), gpu.read(&lse));
    for i in 0..t {
        let r = req[i] as usize;
        for h in 0..heads {
            let qh = &qv[(i * heads + h) * d..(i * heads + h + 1) * d];
            let tag = format!("pool h{heads} d{d} ratio{ratio} sel{top_k:?} row {i} head {h}");
            let have = &got[(i * heads + h) * d..(i * heads + h + 1) * d];
            if lists[i].is_empty() {
                assert!(have.iter().all(|v| *v == 0.0), "{tag}: not zero");
                assert_eq!(got_lse[i * heads + h], f32::NEG_INFINITY, "{tag}: lse");
                continue;
            }
            let s: Vec<f64> = lists[i]
                .iter()
                .map(|&c| {
                    (0..d)
                        .map(|j| f64::from(qh[j]) * f64::from(ents[r][c * d + j]))
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let m = s.iter().copied().fold(f64::MIN, f64::max);
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            let want: Vec<f32> = (0..d)
                .map(|j| {
                    (lists[i]
                        .iter()
                        .zip(&s)
                        .map(|(&c, x)| (x - m).exp() * f64::from(ents[r][c * d + j]))
                        .sum::<f64>()
                        / z) as f32
                })
                .collect();
            let big = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            close_abs(&tag, have, &want, 1.5e-2 * big + 1e-6);
            let want_lse = ((m + z.ln()) * std::f64::consts::LOG2_E) as f32;
            close_abs(
                &format!("{tag} lse"),
                &[got_lse[i * heads + h]],
                &[want_lse],
                2e-3,
            );
        }
    }
}

#[test]
fn pool_attention_over_every_entry() {
    // DeepSeek-V4's HCA: 512-wide entries on the latent walk, split.
    run_pool_lse(64, 512, 128, &[127, 5000, 20000, 60], &[0, 1, 1, 2], None);
    // A narrower pool: the reference walk.
    run_pool_lse(8, 128, 4, &[3, 99, 2, 400], &[0, 0, 1, 1], None);
}

#[test]
fn pool_attention_over_selected_entries() {
    // DeepSeek-V4's CSA: the indexer's top-k of a ratio-4 pool.
    run_pool_lse(64, 512, 4, &[3000, 4100, 1], &[0, 1, 2], Some(512));
    run_pool_lse(
        16,
        512,
        4,
        &[700; 40]
            .iter()
            .enumerate()
            .map(|(i, p)| p + i as i32)
            .collect::<Vec<_>>(),
        &[0; 40],
        Some(64),
    );
    run_pool_lse(4, 192, 4, &[300, 41], &[0, 1], Some(20));
}

/// The block-max ranking (`IndexScore::BlockMax`, MiniMax-M3's indexer):
/// each head scores the `block`-key blocks a row sees by their best `q_h·k`
/// (its own block up to the row), keeps its own, and lists the best
/// `top_k − 1` of the rest with it, ascending, `-1` past the run; a row that
/// sees no more than `top_k` blocks lists them all.
fn run_block_max(
    heads: usize,
    d: usize,
    block: usize,
    top_k: usize,
    qo_lens: &[usize],
    prefix: &[usize],
) {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let ps = 16;
    let kv_lens: Vec<usize> = qo_lens.iter().zip(prefix).map(|(q, p)| q + p).collect();
    let keys: Vec<Vec<f32>> = kv_lens
        .iter()
        .enumerate()
        .map(|(r, &n)| bf16s(n * d, 30 + r as u64, 1.0))
        .collect();
    let p = make_pool(&gpu, &keys, &kv_lens, d, ps);
    let total: usize = qo_lens.iter().sum();
    let qv = bf16s(total * heads * d, 5, 1.0);
    let q = RaggedTensor {
        data: gpu.tensor(&qv, total as u32, (heads * d) as u32, Dtype::Bf16),
        indptr: gpu.column(&indptr_of(qo_lens)),
    };
    let mut sel = gpu.zeros(total as u32, (heads * top_k) as u32, Dtype::I32);
    index::topk_block_max(
        &ctx,
        q,
        &p.pool,
        heads as u32,
        d as u32,
        top_k as u32,
        block as u32,
        &mut sel,
    )
    .unwrap();
    let got = gpu.read_as::<i32>(&sel);
    let mut t = 0;
    for (r, &ql) in qo_lens.iter().enumerate() {
        for i in 0..ql {
            let abs = kv_lens[r] - ql + i;
            let nb = abs / block + 1;
            for h in 0..heads {
                let run = &got[(t * heads + h) * top_k..(t * heads + h + 1) * top_k];
                let tag = format!(
                    "block max h{heads} d{d} block{block} req {r} row {i} head {h} ({nb} blocks)"
                );
                if nb <= top_k {
                    let want: Vec<i32> = (0..top_k as i32)
                        .map(|n| if (n as usize) < nb { n } else { -1 })
                        .collect();
                    assert_eq!(run, &want[..], "{tag}");
                    continue;
                }
                let qh = &qv[(t * heads + h) * d..(t * heads + h + 1) * d];
                let score: Vec<f64> = (0..nb)
                    .map(|b| {
                        (b * block..((b + 1) * block).min(abs + 1))
                            .map(|j| {
                                (0..d)
                                    .map(|e| f64::from(qh[e]) * f64::from(keys[r][j * d + e]))
                                    .sum::<f64>()
                            })
                            .fold(f64::MIN, f64::max)
                    })
                    .collect();
                assert!(
                    run.windows(2).all(|w| w[0] < w[1]),
                    "{tag}: not ascending: {run:?}"
                );
                assert!(
                    run.iter().all(|&b| b >= 0 && (b as usize) < nb),
                    "{tag}: out of range"
                );
                let own = nb - 1;
                assert!(
                    run.contains(&(own as i32)),
                    "{tag}: its own block is dropped"
                );
                let mut picked = vec![false; nb];
                for &b in run {
                    picked[b as usize] = true;
                }
                let lo = run
                    .iter()
                    .filter(|&&b| b as usize != own)
                    .map(|&b| score[b as usize])
                    .fold(f64::MAX, f64::min);
                let hi = (0..own)
                    .filter(|&b| !picked[b])
                    .map(|b| score[b])
                    .fold(f64::MIN, f64::max);
                assert!(
                    lo >= hi - 1e-3 * hi.abs().max(1.0),
                    "{tag}: picked {lo} < left {hi}"
                );
            }
            t += 1;
        }
    }
}

#[test]
fn block_max_ranks_each_heads_blocks() {
    // MiniMax-M3's indexer: four heads of 128, 128-key blocks, 16 a head;
    // decode rows past and short of the budget, and a prefill.
    run_block_max(4, 128, 128, 16, &[1, 1, 1], &[3000, 900, 5000]);
    run_block_max(4, 128, 128, 16, &[300, 7], &[2000, 0]);
    // Small blocks and a narrow run, eight heads.
    run_block_max(8, 64, 8, 5, &[1, 20], &[300, 70]);
}

/// A selection row as `attention.index_topk` writes it: every visible
/// position when they fit the budget, else a budget's worth of distinct ones,
/// ascending either way, `-1` past them.
fn topk_row(visible: usize, top_k: usize, seed: u64) -> Vec<i32> {
    let mut row: Vec<i32> = if visible <= top_k {
        (0..visible as i32).collect()
    } else {
        let mut pool: Vec<i32> = (0..visible as i32).collect();
        let mut s = 0x9e37_79b9u64 ^ seed.wrapping_mul(0x2545_f491);
        for i in 0..top_k {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let j = i + (s as usize) % (visible - i);
            pool.swap(i, j);
        }
        let mut picked = pool[..top_k].to_vec();
        picked.sort_unstable();
        picked
    };
    row.resize(top_k, -1);
    row
}
