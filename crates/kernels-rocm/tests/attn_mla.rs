//! Multi-head latent attention against a host reference: the feed ops
//! (q_b split, latent norm, cache append), the planned decode and causal
//! prefill over the paged latent cache (split and unsplit schedules, with
//! and without a rope plane, 16 to 96 heads), the selected walk over a
//! top-k list, and the reference walk at widths the matrix core does not
//! take.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, KV, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::{mla, plan};
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

/// A paged latent cache: `kv_lens[r]` tokens for request r, pages handed
/// out in reverse so a wrong page walk shows. The host copy is
/// [request][token][rank] and [request][token][rope].
struct Cache {
    pool: KvPool,
    ckv: Vec<Vec<f32>>,
    kpe: Vec<Vec<f32>>,
    page_indptr: Vec<i32>,
    rank: usize,
    rope: usize,
}

#[allow(clippy::too_many_arguments)]
fn pool(
    gpu: &Gpu,
    keys: Tensor,
    values: Tensor,
    page_indices: &[i32],
    page_indptr: &[i32],
    last: &[i32],
    ps: usize,
    seq: usize,
    max_pages: usize,
) -> KvPool {
    KvPool {
        keys,
        values,
        page_indices: gpu.column(page_indices),
        page_indptr: gpu.column(page_indptr),
        last_page_lens: gpu.column(last),
        page_size: ps as i32,
        seq_stride: seq as i64,
        head_stride: seq as i64,
        max_pages_per_request: max_pages as i32,
        pages_in_batch: page_indices.len() as i32,
        live: [0, 0],
        ..KV
    }
}

/// Page tables for `kv_lens` at `ps` tokens a page: (indices, indptr, last
/// page lengths, total pages), pages in reverse order.
fn pages(kv_lens: &[usize], ps: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, usize) {
    let per: Vec<usize> = kv_lens.iter().map(|&n| n.div_ceil(ps).max(1)).collect();
    let total = per.iter().sum::<usize>() + 2;
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
    let last = kv_lens
        .iter()
        .zip(&per)
        .map(|(&n, &p)| (n - (p - 1) * ps) as i32)
        .collect();
    (indices, indptr, last, total)
}

fn cache(gpu: &Gpu, kv_lens: &[usize], rank: usize, rope: usize, ps: usize, seed: u64) -> Cache {
    let (indices, indptr, last, total) = pages(kv_lens, ps);
    let cells = total * ps;
    let mut cp = vec![0f32; cells * rank];
    let mut kp = vec![0f32; cells * rope.max(1)];
    let (mut ckv, mut kpe) = (Vec::new(), Vec::new());
    for (r, &n) in kv_lens.iter().enumerate() {
        let c: Vec<f32> = values(n * rank, seed + r as u64, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let k: Vec<f32> = values(n * rope, seed + 50 + r as u64, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        for t in 0..n {
            let cell = indices[indptr[r] as usize + t / ps] as usize * ps + t % ps;
            cp[cell * rank..(cell + 1) * rank].copy_from_slice(&c[t * rank..(t + 1) * rank]);
            kp[cell * rope..(cell + 1) * rope].copy_from_slice(&k[t * rope..(t + 1) * rope]);
        }
        ckv.push(c);
        kpe.push(k);
    }
    let max_pages = indptr.windows(2).map(|w| w[1] - w[0]).max().unwrap() as usize;
    let keys = gpu.tensor(&cp, cells as u32, rank as u32, Dtype::Bf16);
    let vals = gpu.tensor(&kp, cells as u32, rope.max(1) as u32, Dtype::Bf16);
    Cache {
        pool: pool(
            gpu, keys, vals, &indices, &indptr, &last, ps, rank, max_pages,
        ),
        ckv,
        kpe,
        page_indptr: indptr,
        rank,
        rope,
    }
}

/// One query row's heads against the keys at `visible` positions of
/// request `r`: (o rows [head][rank], lse in log2).
fn reference(
    c: &Cache,
    r: usize,
    qn: &[f32],
    qp: &[f32],
    heads: usize,
    visible: &[usize],
    scale: f64,
) -> (Vec<f32>, Vec<f32>) {
    let (rank, rope) = (c.rank, c.rope);
    let mut out = vec![0f32; heads * rank];
    let mut lse = vec![f32::NEG_INFINITY; heads];
    for h in 0..heads {
        if visible.is_empty() {
            continue;
        }
        let s: Vec<f64> = visible
            .iter()
            .map(|&j| {
                let a: f64 = (0..rank)
                    .map(|d| f64::from(qn[h * rank + d]) * f64::from(c.ckv[r][j * rank + d]))
                    .sum();
                let b: f64 = (0..rope)
                    .map(|d| f64::from(qp[h * rope + d]) * f64::from(c.kpe[r][j * rope + d]))
                    .sum();
                (a + b) * scale
            })
            .collect();
        let m = s.iter().copied().fold(f64::MIN, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let z: f64 = e.iter().sum();
        for d in 0..rank {
            let acc: f64 = visible
                .iter()
                .zip(&e)
                .map(|(&j, w)| w * f64::from(c.ckv[r][j * rank + d]))
                .sum();
            out[h * rank + d] = (acc / z) as f32;
        }
        lse[h] = ((m + z.ln()) * std::f64::consts::LOG2_E) as f32;
    }
    (out, lse)
}

/// Every element within `rel` of the row's largest reference magnitude
/// (bf16 weights and outputs round at about 2^-8 of it), and the row not
/// all near zero, so a walk that drops keys or lands nothing shows.
fn close_rows(what: &str, have: &[f32], want: &[f32], rel: f32) {
    let big = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(big > 0.0 || have.iter().all(|v| *v == 0.0), "{what}: empty");
    close_abs(what, have, want, rel * big + 1e-6);
}

fn workspace(gpu: &Gpu) -> plan::Workspace {
    const BYTES: usize = 128 << 20;
    let int = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    let float = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    plan::Workspace {
        int_ptr: int.ptr,
        int_bytes: BYTES,
        float_ptr: float.ptr,
        float_bytes: BYTES,
    }
}

fn qo_indptr(qo_lens: &[usize]) -> Vec<i32> {
    let mut v = vec![0i32];
    for &l in qo_lens {
        v.push(v.last().unwrap() + l as i32);
    }
    v
}

fn mla_plan(
    gpu: &Gpu,
    c: &Cache,
    qo_lens: &[usize],
    kv_lens: &[usize],
    heads: u32,
    causal: bool,
) -> plan::MlaPlan {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let n = qo_lens.len() as u32;
    let total: usize = qo_lens.iter().sum();
    let kv: Vec<i32> = kv_lens.iter().map(|&k| k as i32).collect();
    let built = plan::plan_mla(
        &qo_indptr(qo_lens),
        &c.page_indptr,
        &kv,
        total as u32,
        n,
        plan::Live {
            requests: n,
            lane_offset: 0,
            row_offset: 0,
            rows: total as u32,
        },
        heads,
        c.rank as u32,
        causal,
        &device,
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    built
}

struct Query {
    q: RaggedTensor,
    q_pe: Tensor,
    qn: Vec<f32>,
    qp: Vec<f32>,
}

fn query(gpu: &Gpu, qo_lens: &[usize], heads: usize, rank: usize, rope: usize, seed: u64) -> Query {
    let total: usize = qo_lens.iter().sum();
    let qn: Vec<f32> = values(total * heads * rank, seed, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let qp: Vec<f32> = values(total * heads * rope, seed + 1, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    Query {
        q: RaggedTensor {
            data: gpu.tensor(&qn, total as u32, (heads * rank) as u32, Dtype::Bf16),
            indptr: gpu.column(&qo_indptr(qo_lens)),
        },
        q_pe: if rope == 0 {
            Tensor::new(0, total as u32, 0, Dtype::Bf16)
        } else {
            gpu.tensor(&qp, total as u32, (heads * rope) as u32, Dtype::Bf16)
        },
        qn,
        qp,
    }
}

/// The planned walk (`mla_decode` / `mla_prefill`) over requests of
/// `qo_lens` new tokens after `prefix` cached ones, against the host.
fn run_planned(heads: usize, rope: usize, ps: usize, qo_lens: &[usize], prefix: &[usize]) {
    run_dense(512, heads, rope, ps, qo_lens, prefix, None, false);
}

/// `mla_decode` / `mla_prefill` at a `rank`-wide latent, reading back
/// through `window` tokens if one is stated, and landing each row's per-head
/// base-2 log-sum-exp when `with_lse`.
#[allow(clippy::too_many_arguments)]
fn run_dense(
    rank: usize,
    heads: usize,
    rope: usize,
    ps: usize,
    qo_lens: &[usize],
    prefix: &[usize],
    window: Option<u32>,
    with_lse: bool,
) {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let kv_lens: Vec<usize> = qo_lens.iter().zip(prefix).map(|(q, p)| q + p).collect();
    let c = cache(&gpu, &kv_lens, rank, rope, ps, 7);
    let decode = qo_lens.iter().all(|&l| l == 1);
    let p = mla_plan(&gpu, &c, qo_lens, &kv_lens, heads as u32, !decode);
    let q = query(&gpu, qo_lens, heads, rank, rope, 3);
    let total: usize = qo_lens.iter().sum();
    let mut o = gpu.zeros(total as u32, (heads * rank) as u32, Dtype::Bf16);
    let mut lse = gpu.zeros(total as u32, heads as u32, Dtype::F32);
    let scale = 1.0 / ((rank + 64) as f32).sqrt();
    let f = if decode {
        mla::attention_decode
    } else {
        mla::attention_prefill
    };
    f(
        &ctx,
        q.q,
        &p,
        q.q_pe,
        &c.pool,
        heads as u32,
        rank as u32,
        scale,
        window,
        &mut o,
        with_lse.then_some(&mut lse),
    )
    .unwrap();
    let got = gpu.read(&o);
    let got_lse = gpu.read(&lse);
    let row = heads * rank;
    let mut t = 0;
    for (r, &ql) in qo_lens.iter().enumerate() {
        for i in 0..ql {
            let pos = kv_lens[r] - ql + i;
            let floor = window.map_or(0, |w| (pos + 1).saturating_sub(w as usize));
            let visible: Vec<usize> = (floor..=pos).collect();
            let (want, want_lse) = reference(
                &c,
                r,
                &q.qn[t * row..(t + 1) * row],
                &q.qp[t * heads * rope..(t + 1) * heads * rope],
                heads,
                &visible,
                f64::from(scale),
            );
            close_rows(
                &format!("mla r{rank} h{heads} rope{rope} ps{ps} w{window:?} req {r} token {i}"),
                &got[t * row..(t + 1) * row],
                &want,
                1.5e-2,
            );
            if with_lse {
                close_abs(
                    &format!("mla lse r{rank} h{heads} w{window:?} req {r} token {i}"),
                    &got_lse[t * heads..(t + 1) * heads],
                    &want_lse,
                    2e-2,
                );
            }
            t += 1;
        }
    }
}

#[test]
fn decode_over_short_and_split_contexts() {
    // GLM-5.3: 64 heads, rope 64; a long request splits across the device.
    run_planned(64, 64, 64, &[1, 1, 1, 1], &[0, 36, 300, 2900]);
}

#[test]
fn decode_ninety_six_heads_spans_two_blocks() {
    // Kimi K3: 96 heads, a cluster of two 64-row blocks per request.
    run_planned(96, 64, 16, &[1, 1, 1], &[4, 129, 1500]);
}

#[test]
fn decode_without_a_rope_plane() {
    // GLM-5.3-Flash: NoPE latent attention.
    run_planned(16, 0, 32, &[1, 1], &[77, 1024]);
}

#[test]
fn the_planned_walk_lands_its_log_sum_exp() {
    // Split and unsplit decode rows, and a causal prefill.
    run_dense(512, 64, 64, 64, &[1, 1, 1], &[0, 300, 2900], None, true);
    run_dense(512, 16, 64, 16, &[5, 70, 1], &[0, 33, 200], None, true);
}

#[test]
fn a_windowed_walk_reads_only_its_window() {
    // The gathered walk at the 512 latent, decode and prefill, windows
    // shorter and longer than the context.
    run_dense(
        512,
        64,
        64,
        32,
        &[1, 1, 1],
        &[0, 300, 2900],
        Some(513),
        true,
    );
    run_dense(512, 16, 64, 16, &[5, 70, 1], &[0, 33, 600], Some(40), true);
    // dots.llm3's sliding layers: a 1024 latent on the reference walk.
    run_dense(1024, 8, 64, 16, &[1, 1], &[20, 600], Some(513), true);
    run_dense(1024, 8, 64, 16, &[9, 3], &[560, 0], Some(513), false);
}

#[test]
fn prefill_is_causal_within_and_across_tiles() {
    run_planned(16, 64, 16, &[5, 70, 1], &[0, 33, 200]);
    run_planned(64, 0, 64, &[3, 9], &[40, 0]);
}

/// The selected walk: each token row lists key positions (some out of its
/// causal range, some −1), as `attention.index_topk` publishes them.
fn run_selected(heads: usize, rope: usize, qo_lens: &[usize], prefix: &[usize], top_k: usize) {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let rank = 512;
    let ps = 32;
    let kv_lens: Vec<usize> = qo_lens.iter().zip(prefix).map(|(q, p)| q + p).collect();
    let c = cache(&gpu, &kv_lens, rank, rope, ps, 9);
    let decode = qo_lens.iter().all(|&l| l == 1);
    let p = mla_plan(&gpu, &c, qo_lens, &kv_lens, heads as u32, !decode);
    let q = query(&gpu, qo_lens, heads, rank, rope, 4);
    let total: usize = qo_lens.iter().sum();
    let mut sel = vec![-1i32; total * top_k];
    let mut lists = Vec::new();
    let mut t = 0;
    for (r, &ql) in qo_lens.iter().enumerate() {
        for i in 0..ql {
            let pos = kv_lens[r] - ql + i;
            sel[t * top_k..(t + 1) * top_k].copy_from_slice(&topk_row(pos + 1, top_k, t as u64));
            lists.push(pos);
            t += 1;
        }
    }
    let selection = gpu.ints_shaped(&sel, total as u32, top_k as u32);
    let mut o = gpu.zeros(total as u32, (heads * rank) as u32, Dtype::Bf16);
    let mut lse = gpu.zeros(total as u32, heads as u32, Dtype::F32);
    let scale = 1.0 / 24.0;
    let f = if decode {
        mla::attention_decode_selected
    } else {
        mla::attention_prefill_selected
    };
    f(
        &ctx,
        q.q,
        &p,
        q.q_pe,
        selection,
        &c.pool,
        heads as u32,
        rank as u32,
        scale,
        &mut o,
        Some(&mut lse),
    )
    .unwrap();
    let got = gpu.read(&o);
    let got_lse = gpu.read(&lse);
    let row = heads * rank;
    let mut t = 0;
    for (r, &ql) in qo_lens.iter().enumerate() {
        for _ in 0..ql {
            let pos = lists[t];
            let visible: Vec<usize> = sel[t * top_k..(t + 1) * top_k]
                .iter()
                .filter(|&&j| j >= 0 && j as usize <= pos)
                .map(|&j| j as usize)
                .collect();
            let (want, want_lse) = reference(
                &c,
                r,
                &q.qn[t * row..(t + 1) * row],
                &q.qp[t * heads * rope..(t + 1) * heads * rope],
                heads,
                &visible,
                f64::from(scale),
            );
            close_rows(
                &format!("selected h{heads} rope{rope} row {t}"),
                &got[t * row..(t + 1) * row],
                &want,
                1.5e-2,
            );
            close_abs(
                &format!("selected lse h{heads} rope{rope} row {t}"),
                &got_lse[t * heads..(t + 1) * heads],
                &want_lse,
                2e-2,
            );
            t += 1;
        }
    }
}

#[test]
fn selected_decode_walks_its_list_in_splits() {
    run_selected(64, 64, &[1, 1], &[3000, 70], 2048);
    run_selected(16, 0, &[1], &[600], 300);
}

#[test]
fn selected_prefill_skips_entries_past_the_row() {
    run_selected(64, 64, &[6, 2], &[10, 400], 96);
}

#[test]
fn a_long_unsplit_list_refills_the_slot_table() {
    // Enough rows that each takes one split: 22 tiles a block, the slot
    // table refilled past tiles 8 and 16.
    run_selected(2, 64, &[700], &[40], 700);
}

#[test]
fn the_reference_walk_takes_other_widths() {
    // A 128 latent and a 32 rope: the matrix core walk is built at 512.
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (heads, rank, rope) = (6usize, 128usize, 32usize);
    let qo_lens = [3usize, 1];
    let kv_lens = [10usize, 75];
    let c = cache(&gpu, &kv_lens, rank, rope, 16, 13);
    let p = mla_plan(&gpu, &c, &qo_lens, &kv_lens, heads as u32, true);
    let q = query(&gpu, &qo_lens, heads, rank, rope, 5);
    let mut o = gpu.zeros(4, (heads * rank) as u32, Dtype::Bf16);
    let scale = 0.11;
    mla::attention_prefill(
        &ctx,
        q.q,
        &p,
        q.q_pe,
        &c.pool,
        heads as u32,
        rank as u32,
        scale,
        None,
        &mut o,
        None,
    )
    .unwrap();
    let got = gpu.read(&o);
    let row = heads * rank;
    let mut t = 0;
    for (r, &ql) in qo_lens.iter().enumerate() {
        for i in 0..ql {
            let pos = kv_lens[r] - ql + i;
            let visible: Vec<usize> = (0..=pos).collect();
            let (want, _) = reference(
                &c,
                r,
                &q.qn[t * row..(t + 1) * row],
                &q.qp[t * heads * rope..(t + 1) * heads * rope],
                heads,
                &visible,
                scale.into(),
            );
            close_rows("naive", &got[t * row..(t + 1) * row], &want, 1e-2);
            t += 1;
        }
    }
}

#[test]
fn feed_ops_split_norm_and_append() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // q_b → q_nope, q_pe, under a window of rows 1..3 of 4.
    let (rows, heads, nope, rope) = (4usize, 3usize, 192usize, 64usize);
    let per = nope + rope;
    let qb: Vec<f32> = values(rows * heads * per, 1, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let q_b = gpu.tensor(&qb, rows as u32, (heads * per) as u32, Dtype::Bf16);
    let mut qn = gpu.zeros(rows as u32, (heads * nope) as u32, Dtype::Bf16);
    let mut qp = gpu.zeros(rows as u32, (heads * rope) as u32, Dtype::Bf16);
    let win = gpu.ints(&[2, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    mla::split_q_b(
        &ctx,
        q_b,
        heads as u32,
        nope as u32,
        rope as u32,
        &mut qn,
        &mut qp,
    )
    .unwrap();
    ctx.disarm_stage();
    let (gn, gp) = (gpu.read(&qn), gpu.read(&qp));
    for n in 0..rows {
        for h in 0..heads {
            for d in 0..per {
                let live = (1..3).contains(&n);
                let want = if live {
                    qb[(n * heads + h) * per + d]
                } else {
                    0.0
                };
                let have = if d < nope {
                    gn[(n * heads + h) * nope + d]
                } else {
                    gp[(n * heads + h) * rope + d - nope]
                };
                assert_eq!(have, want, "split row {n} head {h} col {d}");
            }
        }
    }

    // kv_a → normed latent + rope copy, from a row wider than both.
    let (rows, rank, rope, stride) = (5usize, 512usize, 64usize, 600usize);
    let kva: Vec<f32> = values(rows * stride, 2, 2.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let w: Vec<f32> = values(rank, 3, 1.0).into_iter().map(round_bf16).collect();
    let kv_a = gpu.tensor(&kva, rows as u32, stride as u32, Dtype::Bf16);
    let weight = gpu.tensor(&w, 1, rank as u32, Dtype::Bf16);
    let mut kv_c = gpu.zeros(rows as u32, rank as u32, Dtype::Bf16);
    let mut k_pe = gpu.zeros(rows as u32, rope as u32, Dtype::Bf16);
    mla::latents(&ctx, kv_a, weight, 1e-6, rank as u32, &mut kv_c, &mut k_pe).unwrap();
    let (gc, gk) = (gpu.read(&kv_c), gpu.read(&k_pe));
    for n in 0..rows {
        let x = &kva[n * stride..n * stride + rank];
        let ms = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / rank as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        let want: Vec<f32> = x
            .iter()
            .zip(&w)
            .map(|(v, w)| (f64::from(*v) * inv * f64::from(*w)) as f32)
            .collect();
        close_bf16("latent", &gc[n * rank..(n + 1) * rank], &want, 1.0);
        assert_eq!(
            &gk[n * rope..(n + 1) * rope],
            &kva[n * stride + rank..n * stride + rank + rope]
        );
    }

    // The append: request 0 has 3 cached + 2 new, request 1 has 0 + 3.
    let (rank, rope, ps) = (512usize, 64usize, 4usize);
    let kv_lens = [5usize, 3];
    let (indices, indptr, last, total) = pages(&kv_lens, ps);
    let cells = total * ps;
    let keys = gpu.zeros(cells as u32, rank as u32, Dtype::Bf16);
    let vals = gpu.zeros(cells as u32, rope as u32, Dtype::Bf16);
    let pl = pool(&gpu, keys, vals, &indices, &indptr, &last, ps, rank, 2);
    let new = 5usize;
    let ckv: Vec<f32> = values(new * rank, 4, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let kpe: Vec<f32> = values(new * rope, 5, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let rag = RaggedTensor {
        data: gpu.tensor(&ckv, new as u32, rank as u32, Dtype::Bf16),
        indptr: gpu.column(&[0, 2, 5]),
    };
    let kp = gpu.tensor(&kpe, new as u32, rope as u32, Dtype::Bf16);
    mla::kv_append(&ctx, rag, kp, &pl, Tensor::ABSENT, Tensor::ABSENT).unwrap();
    let (gk, gv) = (gpu.read(&pl.keys), gpu.read(&pl.values));
    let land = [(0usize, 3usize), (0, 4), (1, 0), (1, 1), (1, 2)];
    for (t, &(r, pos)) in land.iter().enumerate() {
        let cell = indices[indptr[r] as usize + pos / ps] as usize * ps + pos % ps;
        assert_eq!(
            &gk[cell * rank..(cell + 1) * rank],
            &ckv[t * rank..(t + 1) * rank]
        );
        assert_eq!(
            &gv[cell * rope..(cell + 1) * rope],
            &kpe[t * rope..(t + 1) * rope]
        );
    }
}

#[test]
fn selected_decode_under_a_window() {
    // Three launch rows, the staged window live on rows 1..3: the first
    // stays untouched, and the live rows' requests count from the window.
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (heads, rank, rope, top_k) = (16usize, 512usize, 64usize, 512usize);
    let kv_lens = [700usize, 90];
    let c = cache(&gpu, &kv_lens, rank, rope, 32, 21);
    let p = mla_plan(&gpu, &c, &[1, 1], &kv_lens, heads as u32, false);
    let qn: Vec<f32> = values(3 * heads * rank, 5, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let qp: Vec<f32> = values(3 * heads * rope, 6, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let q = RaggedTensor {
        data: gpu.tensor(&qn, 3, (heads * rank) as u32, Dtype::Bf16),
        indptr: gpu.column(&[0, 1, 2]),
    };
    let q_pe = gpu.tensor(&qp, 3, (heads * rope) as u32, Dtype::Bf16);
    // (the dead row's list never read)
    let sel: Vec<i32> = [800usize, kv_lens[0], kv_lens[1]]
        .iter()
        .enumerate()
        .flat_map(|(i, &visible)| topk_row(visible, top_k, i as u64))
        .collect();
    let selection = gpu.ints_shaped(&sel, 3, top_k as u32);
    let mut o = gpu.zeros(3, (heads * rank) as u32, Dtype::Bf16);
    let win = gpu.ints(&[2, 1, 2, 0]);
    ctx.arm_stage(win.ptr);
    mla::attention_decode_selected(
        &ctx,
        q,
        &p,
        q_pe,
        selection,
        &c.pool,
        heads as u32,
        rank as u32,
        0.05,
        &mut o,
        None,
    )
    .unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&o);
    let row = heads * rank;
    assert!(
        got[..row].iter().all(|v| *v == 0.0),
        "the dead row was written"
    );
    for (r, &kv) in kv_lens.iter().enumerate() {
        let t = r + 1;
        let pos = kv - 1;
        let visible: Vec<usize> = sel[t * top_k..(t + 1) * top_k]
            .iter()
            .filter(|&&j| j >= 0 && j as usize <= pos)
            .map(|&j| j as usize)
            .collect();
        let (want, _) = reference(
            &c,
            r,
            &qn[t * row..(t + 1) * row],
            &qp[t * heads * rope..(t + 1) * heads * rope],
            heads,
            &visible,
            0.05,
        );
        close_rows(
            &format!("windowed row {t}"),
            &got[t * row..(t + 1) * row],
            &want,
            1.5e-2,
        );
    }
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

#[test]
fn absorb_into_and_out_of_the_latent() {
    // kv_b per head: nope rows (the keys' up-projection) then v rows, each
    // rank wide. q_nope · W_k gives the latent query; latent · W_vᵀ the
    // head's output.
    let Some(gpu) = Gpu::open() else { return };
    // rocBLAS's strided batch, then the native kernel over a decode's few
    // rows and rows past one tile ending mid-tile.
    for (native, tokens) in [(false, 3usize), (true, 3), (true, 130)] {
        let mut ctx = gpu.ctx();
        let mut tuning = ctx.tuning();
        if native {
            tuning.mla_absorb_q_until = u32::MAX;
            tuning.mla_absorb_out_until = u32::MAX;
        }
        ctx.set_tuning(tuning);
        let (heads, nope, v, rank) = (4usize, 192usize, 256usize, 512usize);
        let kvb: Vec<f32> = values(heads * (nope + v) * rank, 31, 0.1)
            .into_iter()
            .map(round_bf16)
            .collect();
        let qn: Vec<f32> = values(tokens * heads * nope, 32, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let lat: Vec<f32> = values(tokens * heads * rank, 33, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let kv_b = gpu.tensor(&kvb, (heads * (nope + v)) as u32, rank as u32, Dtype::Bf16);
        let mut q_latent = gpu.zeros(tokens as u32, (heads * rank) as u32, Dtype::Bf16);
        mla::absorb_q(
            &ctx,
            gpu.tensor(&qn, tokens as u32, (heads * nope) as u32, Dtype::Bf16),
            kv_b,
            heads as u32,
            rank as u32,
            nope as u32,
            v as u32,
            &mut q_latent,
        )
        .unwrap();
        let mut o = gpu.zeros(tokens as u32, (heads * v) as u32, Dtype::Bf16);
        mla::absorb_out(
            &ctx,
            gpu.tensor(&lat, tokens as u32, (heads * rank) as u32, Dtype::Bf16),
            kv_b,
            heads as u32,
            rank as u32,
            v as u32,
            nope as u32,
            &mut o,
        )
        .unwrap();
        let (gq, go) = (gpu.read(&q_latent), gpu.read(&o));
        let w =
            |h: usize, row: usize, col: usize| f64::from(kvb[(h * (nope + v) + row) * rank + col]);
        for t in 0..tokens {
            for h in 0..heads {
                let want: Vec<f32> = (0..rank)
                    .map(|i| {
                        (0..nope)
                            .map(|k| f64::from(qn[(t * heads + h) * nope + k]) * w(h, k, i))
                            .sum::<f64>() as f32
                    })
                    .collect();
                let at = (t * heads + h) * rank;
                close_rows("absorb_q", &gq[at..at + rank], &want, 1e-2);
                let want: Vec<f32> = (0..v)
                    .map(|j| {
                        (0..rank)
                            .map(|r| f64::from(lat[(t * heads + h) * rank + r]) * w(h, nope + j, r))
                            .sum::<f64>() as f32
                    })
                    .collect();
                let at = (t * heads + h) * v;
                close_rows("absorb_out", &go[at..at + v], &want, 1e-2);
            }
        }
    }
}

#[test]
fn latents_rope_turns_the_rope_half() {
    // The norm as `mla_latents`, then the rope half turned (pair i with
    // i + 32, theta^(−i/32) radians a position).
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (rows, rank, rope, theta) = (3usize, 512usize, 64usize, 10000f32);
    let pos = [0i32, 5, 1000];
    let kva: Vec<f32> = values(rows * (rank + rope), 41, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let w = vec![1f32; rank];
    let mut kv_c = gpu.zeros(rows as u32, rank as u32, Dtype::Bf16);
    let mut k_pe = gpu.zeros(rows as u32, rope as u32, Dtype::Bf16);
    mla::latents_rope(
        &ctx,
        gpu.tensor(&kva, rows as u32, (rank + rope) as u32, Dtype::Bf16),
        gpu.column(&pos),
        gpu.tensor(&w, 1, rank as u32, Dtype::Bf16),
        1e-6,
        rank as u32,
        rope as u32,
        theta,
        &mut kv_c,
        &mut k_pe,
    )
    .unwrap();
    let got = gpu.read(&k_pe);
    let half = rope / 2;
    for n in 0..rows {
        let x = &kva[n * (rank + rope) + rank..(n + 1) * (rank + rope)];
        let mut want = vec![0f32; rope];
        for i in 0..half {
            let freq = f64::from(theta).powf(-(i as f64) / half as f64);
            let (s, c) = (f64::from(pos[n]) * freq).sin_cos();
            let (a, b) = (f64::from(x[i]), f64::from(x[i + half]));
            want[i] = (a * c - b * s) as f32;
            want[i + half] = (b * c + a * s) as f32;
        }
        close_abs("rope half", &got[n * rope..(n + 1) * rope], &want, 8e-3);
    }
}
