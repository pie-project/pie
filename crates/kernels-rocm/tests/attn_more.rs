//! The attention auxiliaries against host references: the logit soft cap,
//! the sink rescale, the lse merge, dense bidirectional attention over
//! segments, attention over selected key blocks, the selector walk, score
//! capture, the block dynamic conv, the hashed n-gram ids, and the fused
//! decode qkv epilogue. Ragged shapes throughout, and the staged window
//! where a launcher reads one.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, KV, STATE, bf16s, close_abs, close_bf16, indptr_of, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, plan, selected, selector};
use kernels_rocm::{KvPool, RaggedTensor, RecurrentPool};

/// A bf16 NHD pool and the host keys and values it holds, [request][token]
/// [head][d].
struct Cache {
    pool: KvPool,
    keys: Vec<Vec<f32>>,
    vals: Vec<Vec<f32>>,
    page_indptr: Vec<i32>,
}

/// A bf16 NHD paged cache holding `kv_lens[r]` tokens for request r, pages
/// handed out in reverse.
fn paged_cache(
    gpu: &Gpu,
    kv_heads: usize,
    d: usize,
    ps: usize,
    kv_lens: &[u32],
    seed: u64,
) -> Cache {
    let per: Vec<usize> = kv_lens
        .iter()
        .map(|&n| (n as usize).div_ceil(ps).max(1))
        .collect();
    let total = per.iter().sum::<usize>() + 2;
    let (mut indices, mut indptr) = (Vec::new(), vec![0i32]);
    for &p in &per {
        for _ in 0..p {
            indices.push((total - 1 - indices.len()) as i32);
        }
        indptr.push(indices.len() as i32);
    }
    let row = kv_heads * d;
    let (mut kp, mut vp) = (vec![0f32; total * ps * row], vec![0f32; total * ps * row]);
    let (mut keys, mut vals) = (Vec::new(), Vec::new());
    for (r, &n) in kv_lens.iter().enumerate() {
        let k = bf16s(n as usize * row, seed + r as u64, 1.0);
        let v = bf16s(n as usize * row, seed + 50 + r as u64, 1.0);
        for t in 0..n as usize {
            let slot = indices[indptr[r] as usize + t / ps] as usize * ps + t % ps;
            kp[slot * row..(slot + 1) * row].copy_from_slice(&k[t * row..(t + 1) * row]);
            vp[slot * row..(slot + 1) * row].copy_from_slice(&v[t * row..(t + 1) * row]);
        }
        keys.push(k);
        vals.push(v);
    }
    let last: Vec<i32> = kv_lens
        .iter()
        .zip(&per)
        .map(|(&n, &p)| (n as usize - (p - 1) * ps) as i32)
        .collect();
    let rows = (total * ps) as u32;
    let pool = KvPool {
        keys: gpu.tensor(&kp, rows, row as u32, Dtype::Bf16),
        values: gpu.tensor(&vp, rows, row as u32, Dtype::Bf16),
        page_indices: gpu.column(&indices),
        page_indptr: gpu.column(&indptr),
        last_page_lens: gpu.column(&last),
        page_size: ps as i32,
        seq_stride: row as i64,
        head_stride: d as i64,
        max_pages_per_request: *per.iter().max().unwrap() as i32,
        pages_in_batch: indices.len() as i32,
        ..KV
    };
    Cache {
        pool,
        keys,
        vals,
        page_indptr: indptr,
    }
}

/// softmax(q · kᵀ · scale) · v over the keys `keys` names, in f64.
fn attend(
    q: &[f32],
    key: impl Fn(usize) -> (Vec<f32>, Vec<f32>),
    keys: &[usize],
    scale: f64,
) -> Vec<f32> {
    let d = q.len();
    if keys.is_empty() {
        return vec![0.0; d];
    }
    let kv: Vec<_> = keys.iter().map(|&j| key(j)).collect();
    let s: Vec<f64> = kv
        .iter()
        .map(|(k, _)| {
            (0..d)
                .map(|i| f64::from(q[i]) * f64::from(k[i]))
                .sum::<f64>()
                * scale
        })
        .collect();
    let m = s.iter().copied().fold(f64::MIN, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    (0..d)
        .map(|i| {
            (kv.iter()
                .zip(&e)
                .map(|((_, v), w)| w * f64::from(v[i]))
                .sum::<f64>()
                / z) as f32
        })
        .collect()
}

// --------------------------------------------------------- elementwise --

#[test]
fn logit_softcap_caps_every_logit() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let xs = bf16s(3 * 1001, 1, 90.0);
    let mut x = gpu.tensor(&xs, 3, 1001, Dtype::Bf16);
    attn::logit_softcap(&ctx, &mut x, 30.0).unwrap();
    let inv = 1.0f32 / 30.0;
    let want: Vec<f32> = xs.iter().map(|&v| 30.0 * (v * inv).tanh()).collect();
    close_bf16("logit_softcap", &gpu.read(&x), &want, 1.0);
}

#[test]
fn sink_rescales_by_the_sinks_share() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (heads, d) in [(4u32, 64u32), (3, 20), (2, 200)] {
        let rows = 5u32;
        let os = bf16s((rows * heads * d) as usize, 2, 1.0);
        let mut lses = values((rows * heads) as usize, 3, 6.0);
        lses[1] = f32::NEG_INFINITY;
        let sinks = bf16s(heads as usize, 4, 2.0);
        let lse = gpu.tensor(&lses, rows, heads, Dtype::F32);
        let sink = gpu.tensor(&sinks, 1, heads, Dtype::Bf16);
        let ratio = |r: usize, h: usize| {
            let l = lses[r * heads as usize + h];
            if l.is_finite() {
                1.0 / (1.0 + (-(l * std::f32::consts::LN_2 - sinks[h])).exp())
            } else {
                1.0
            }
        };
        let want = |live: &dyn Fn(usize) -> bool| -> Vec<f32> {
            os.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let (r, h) = (
                        i / (heads * d) as usize,
                        i % (heads * d) as usize / d as usize,
                    );
                    if live(r) { v * ratio(r, h) } else { v }
                })
                .collect()
        };

        let mut o = gpu.tensor(&os, rows, heads * d, Dtype::Bf16);
        attn::sink(&ctx, &mut o, lse, sink, d).unwrap();
        close_bf16("sink", &gpu.read(&o), &want(&|_| true), 1.0);

        // The sinks in f32, as a checkpoint may keep them (Hy4).
        let sink32 = gpu.tensor(&sinks, 1, heads, Dtype::F32);
        let mut o = gpu.tensor(&os, rows, heads * d, Dtype::Bf16);
        attn::sink(&ctx, &mut o, lse, sink32, d).unwrap();
        close_bf16("f32 sink", &gpu.read(&o), &want(&|_| true), 1.0);

        // Staged rows 1..4 of the 5.
        let mut o = gpu.tensor(&os, rows, heads * d, Dtype::Bf16);
        let win = gpu.ints(&[3, 1, 0, 0]);
        ctx.arm_stage(win.ptr);
        attn::sink(&ctx, &mut o, lse, sink, d).unwrap();
        ctx.disarm_stage();
        close_bf16(
            "windowed sink",
            &gpu.read(&o),
            &want(&|r| (1..4).contains(&r)),
            1.0,
        );
    }
}

#[test]
fn merge_lse_combines_two_partials() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (n, heads, d) = (7u32, 3u32, 72u32);
    let (o1s, o2s) = (
        bf16s((n * heads * d) as usize, 5, 1.0),
        bf16s((n * heads * d) as usize, 6, 1.0),
    );
    let mut l1s = values((n * heads) as usize, 7, 8.0);
    let mut l2s = values((n * heads) as usize, 8, 8.0);
    l1s[2] = f32::NEG_INFINITY;
    l2s[4] = f32::NEG_INFINITY;
    l1s[5] = f32::NEG_INFINITY;
    l2s[5] = f32::NEG_INFINITY;
    let (o1, o2) = (
        gpu.tensor(&o1s, n, heads * d, Dtype::Bf16),
        gpu.tensor(&o2s, n, heads * d, Dtype::Bf16),
    );
    let (lse1, lse2) = (
        gpu.tensor(&l1s, n, heads, Dtype::F32),
        gpu.tensor(&l2s, n, heads, Dtype::F32),
    );
    let mut o = gpu.zeros(n, heads * d, Dtype::Bf16);
    let mut lse = gpu.zeros(n, heads, Dtype::F32);
    attn::merge_lse(&ctx, o1, lse1, o2, lse2, heads, d, &mut o, &mut lse).unwrap();
    let (mut want, mut want_lse) = (vec![0f32; o1s.len()], vec![0f32; l1s.len()]);
    for cell in 0..(n * heads) as usize {
        let (a, b) = (l1s[cell], l2s[cell]);
        let span = cell * d as usize..(cell + 1) * d as usize;
        if !b.is_finite() {
            want[span.clone()].copy_from_slice(&o1s[span]);
            want_lse[cell] = a;
        } else if !a.is_finite() {
            want[span.clone()].copy_from_slice(&o2s[span]);
            want_lse[cell] = b;
        } else {
            let big = a.max(b);
            let (w1, w2) = ((a - big).exp2(), (b - big).exp2());
            for i in span {
                want[i] = (o1s[i] * w1 + o2s[i] * w2) * (1.0 / (w1 + w2));
            }
            want_lse[cell] = big + (w1 + w2).log2();
        }
    }
    close_bf16("merge_lse", &gpu.read(&o), &want, 1.0);
    let got_lse = gpu.read(&lse);
    for (i, (h, w)) in got_lse.iter().zip(&want_lse).enumerate() {
        assert!(
            (h.is_infinite() && w.is_infinite()) || (h - w).abs() <= 1e-5 * w.abs().max(1.0),
            "merge lse[{i}] = {h}, want {w}"
        );
    }
}

// --------------------------------------------------------------- dense --

#[test]
fn dense_attends_inside_each_segment() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // Rows 0..3 and 200..210 belong to no segment; [40, 40) is empty.
    let segments = [3i32, 40, 40, 150, 200];
    let rows = 210u32;
    for (hq, hk, d) in [
        (4u32, 2u32, 72u32),
        (2, 2, 64),
        (2, 1, 60),
        (2, 2, 128),
        (1, 1, 256),
        (2, 2, 90),
    ] {
        let (qs, ks, vs) = (
            bf16s((rows * hq * d) as usize, 9, 1.0),
            bf16s((rows * hk * d) as usize, 10, 1.0),
            bf16s((rows * hk * d) as usize, 11, 1.0),
        );
        let q = gpu.tensor(&qs, rows, hq * d, Dtype::Bf16);
        let k = gpu.tensor(&ks, rows, hk * d, Dtype::Bf16);
        let v = gpu.tensor(&vs, rows, hk * d, Dtype::Bf16);
        let seg = gpu.column(&segments);
        let scale = 1.0 / (d as f32).sqrt();
        let mut o = gpu.zeros(rows, hq * d, Dtype::Bf16);
        kernels_rocm::attn_dense::bidirectional(&ctx, q, k, v, seg, d, scale, &mut o).unwrap();
        let got = gpu.read(&o);
        let (d_, g) = (d as usize, (hq / hk) as usize);
        for r in 0..rows as usize {
            let span = segments
                .windows(2)
                .find(|s| (s[0] as usize..s[1] as usize).contains(&r))
                .map_or(0..0, |s| s[0] as usize..s[1] as usize);
            let keys: Vec<usize> = span.collect();
            for h in 0..hq as usize {
                let kh = h / g;
                let at = |j: usize| (j * hk as usize + kh) * d_;
                let want = attend(
                    &qs[(r * hq as usize + h) * d_..(r * hq as usize + h + 1) * d_],
                    |j| {
                        (
                            ks[at(j)..at(j) + d_].to_vec(),
                            vs[at(j)..at(j) + d_].to_vec(),
                        )
                    },
                    &keys,
                    f64::from(scale),
                );
                close_abs(
                    &format!("dense d{d} g{g} row {r} head {h}"),
                    &got[(r * hq as usize + h) * d_..(r * hq as usize + h + 1) * d_],
                    &want,
                    1e-2,
                );
            }
        }
    }
}

// ------------------------------------------------------------ selected --

#[derive(Clone, Copy, PartialEq)]
enum Walk {
    Prefill,
    Decode,
    Tiled,
}
use Walk::{Decode, Prefill, Tiled};

/// `((hq, hk, d), (page, ratio, top_k), fresh, prefix, walk, per_head, seed)`.
type Sel<'a> = (
    (u32, u32, u32),
    (usize, u32, u32),
    &'a [u32],
    &'a [u32],
    Walk,
    bool,
    u64,
);

#[test]
fn selected_attention_reads_the_named_blocks_and_the_open_one() {
    let Some(gpu) = Gpu::open() else { return };
    let decode = vec![1u32; 128];
    let decode_prefix: Vec<u32> = (0..128u32).map(|i| (i * 37 + 11) % 200).collect();
    #[rustfmt::skip]
    let cases: &[Sel] = &[
        // Vector walk; enough rows that a block answers two heads.
        ((4, 2, 64), (16, 4, 5), &[1, 6, 3], &[40, 0, 70], Prefill, false, 30),
        ((2, 1, 96), (16, 4, 5), &[150, 1, 160], &[40, 0, 70], Prefill, false, 31),
        // Qwen3.8-Flash-Next's decode shape, and entries past one LDS window.
        ((24, 2, 256), (16, 4, 32), &decode, &decode_prefix, Decode, false, 70),
        ((8, 1, 128), (16, 4, 600), &[1, 2], &[2600, 2399], Prefill, false, 91),
        // The tiled causal walk: inside the budget, past it, across tiles.
        ((24, 2, 256), (16, 4, 16), &[130, 3, 61], &[300, 900, 41], Tiled, false, 93),
        // MiniMax-M3's per-kv-head selections: vector, matrix-core, tiled.
        ((16, 4, 96), (16, 8, 4), &[1, 1, 1], &[90, 7, 300], Decode, true, 101),
        ((16, 4, 128), (16, 8, 4), &[9, 3], &[40, 200], Prefill, true, 102),
        ((64, 4, 128), (64, 128, 16), &[1, 1], &[2900, 1500], Decode, true, 103),
        ((32, 2, 128), (16, 8, 8), &[130, 3, 61], &[300, 900, 41], Tiled, true, 104),
    ];
    for &case in cases {
        check_selected(&gpu, case);
    }
}

/// `per_head`: a run of `top_k` blocks for each kv head, each head's
/// queries reading their own.
fn check_selected(gpu: &Gpu, case: Sel) {
    let ((hq, hk, d), (ps, ratio, top_k), fresh, prefix, walk, per_head, seed) = case;
    let runs = if per_head { hk } else { 1 };
    let ctx = gpu.ctx();
    let kv_lens: Vec<u32> = fresh.iter().zip(prefix).map(|(f, p)| f + p).collect();
    let Cache {
        pool,
        keys,
        vals,
        page_indptr,
    } = paged_cache(gpu, hk as usize, d as usize, ps, &kv_lens, seed);
    let qo = indptr_of(fresh);
    let total = *qo.last().unwrap() as usize;
    let qs = bf16s(total * (hq * d) as usize, seed + 7, 1.0);
    // A row's selection: distinct block ids, some past its causal end and
    // some -1.
    let mut sel = Vec::new();
    for t in 0..total {
        for kh in 0..runs as i32 {
            for n in 0..top_k as i32 {
                sel.push(match (t as i32 + n + kh) % 4 {
                    0 => -1,
                    _ => (n * 3 + t as i32 + 5 * kh) % (4 * top_k as i32).max(20),
                });
            }
        }
    }
    // Distinct: (n·3 + t) mod M repeats only when 3·Δn ≡ 0 (mod M).
    let q = RaggedTensor {
        data: gpu.tensor(&qs, total as u32, hq * d, Dtype::Bf16),
        indptr: gpu.column(&qo),
    };
    let selection = gpu.ints_shaped(&sel, total as u32, top_k * runs);
    let scale = 1.0 / (d as f32).sqrt();
    let mut o = gpu.zeros(total as u32, hq * d, Dtype::Bf16);
    if walk == Tiled {
        let device = plan::Device::probe(&ctx).unwrap();
        let int = gpu.zeros(1, 16 << 20, Dtype::F32);
        let float = gpu.zeros(1, 16 << 20, Dtype::F32);
        let n = fresh.len() as u32;
        let built = plan::plan_prefill(
            &qo,
            &page_indptr,
            &kv_lens.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            total as u32,
            plan::Shape {
                num_requests: n,
                lane_offset: 0,
                num_q_heads: hq,
                num_kv_heads: hk,
                head_dim: d,
                page_size: ps as u32,
                hnd_layout: false,
            },
            plan::Live {
                requests: n,
                lane_offset: 0,
                row_offset: 0,
                rows: total as u32,
            },
            None,
            true,
            false,
            None,
            &device,
            plan::Toggles::default(),
            plan::Workspace {
                int_ptr: int.ptr,
                int_bytes: 64 << 20,
                float_ptr: float.ptr,
                float_bytes: 64 << 20,
            },
        )
        .unwrap();
        built.stage(&ctx).unwrap();
        let took = attn::prefill_selected_tiled(
            &ctx, q, &built, selection, &pool, None, d, hk, scale, ratio, per_head, &mut o,
        )
        .unwrap();
        assert!(took, "the tiled selected prefill refused d{d} {hq}/{hk}");
        gpu.sync();
    } else if walk == Decode {
        selected::attention_decode_selected(
            &ctx, q, selection, &pool, None, d, scale, ratio, per_head, &mut o,
        )
        .unwrap();
    } else {
        selected::attention_prefill_selected(
            &ctx, q, selection, &pool, None, d, hk, scale, ratio, per_head, &mut o,
        )
        .unwrap();
    }
    let got = gpu.read(&o);
    let (d_, g, ratio_) = (d as usize, (hq / hk) as usize, ratio as usize);
    for r in 0..fresh.len() {
        for i in 0..fresh[r] as usize {
            let t = qo[r] as usize + i;
            let j_end = prefix[r] as usize + i + 1;
            let nblocks = j_end / ratio_;
            let keys_of = |run: usize| {
                let mut picked: Vec<usize> = Vec::new();
                let base = (t * runs as usize + run) * top_k as usize;
                for n in 0..(top_k as usize).min(nblocks) {
                    let c = sel[base + n];
                    if c >= 0 && (c as usize) < nblocks {
                        assert!(
                            !picked.contains(&(c as usize)),
                            "the test's selection repeats"
                        );
                        picked.push(c as usize);
                    }
                }
                let mut ks_: Vec<usize> = picked
                    .iter()
                    .flat_map(|&c| c * ratio_..(c + 1) * ratio_)
                    .collect();
                ks_.extend(nblocks * ratio_..j_end);
                ks_
            };
            for h in 0..hq as usize {
                let kh = h / g;
                let ks_ = keys_of(if per_head { kh } else { 0 });
                let at = |j: usize| (j * hk as usize + kh) * d_;
                let want = attend(
                    &qs[(t * hq as usize + h) * d_..(t * hq as usize + h + 1) * d_],
                    |j| {
                        (
                            keys[r][at(j)..at(j) + d_].to_vec(),
                            vals[r][at(j)..at(j) + d_].to_vec(),
                        )
                    },
                    &ks_,
                    f64::from(scale),
                );
                close_abs(
                    &format!("selected d{d} g{g} per_head {per_head} row {t} head {h}"),
                    &got[(t * hq as usize + h) * d_..(t * hq as usize + h + 1) * d_],
                    &want,
                    1e-2,
                );
            }
        }
    }
}

// ------------------------------------------------------------ selector --

#[test]
fn selector_walk_chains_each_rows_pick() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (k, rank, vocab) = (5u32, 24u32, 50u32);
    let lens = [4u32, 0, 5, 1];
    let indptr = indptr_of(&lens);
    let rows = *indptr.last().unwrap() as usize;
    let mut cand: Vec<i32> = (0..rows * k as usize)
        .map(|i| ((i * 7 + 3) % vocab as usize) as i32)
        .collect();
    cand[3] = -1;
    let tokens: Vec<i32> = (0..rows)
        .map(|i| ((i * 11 + 1) % vocab as usize) as i32)
        .collect();
    let unary = values(rows * k as usize, 13, 0.5);
    let pred = bf16s((vocab * rank) as usize, 14, 1.0);
    let succ = bf16s((vocab * rank) as usize, 15, 1.0);
    let hp = bf16s(rows * rank as usize, 16, 1.0);
    for (first, with_hp) in [(0u32, false), (1, true), (1, false), (0, true)] {
        let c = RaggedTensor {
            data: gpu.ints_shaped(&cand, rows as u32, k),
            indptr: gpu.column(&indptr),
        };
        let un = gpu.tensor(&unary, rows as u32, k, Dtype::F32);
        let hpt = with_hp.then(|| gpu.tensor(&hp, rows as u32, rank, Dtype::Bf16));
        let tok = gpu.column(&tokens);
        let (pr, su) = (
            gpu.tensor(&pred, vocab, rank, Dtype::Bf16),
            gpu.tensor(&succ, vocab, rank, Dtype::Bf16),
        );
        let mut picks = gpu.upload(&vec![-7i32; rows], rows as u32, 1, Dtype::I32);
        selector::walk(&ctx, c, un, hpt, tok, pr, su, first, &mut picks).unwrap();
        let got = gpu.read_as::<i32>(&picks);

        let mut want = vec![-7i32; rows];
        for r in 0..lens.len() {
            let (b, e) = (indptr[r] as usize, indptr[r + 1] as usize);
            if e <= b {
                continue;
            }
            if first > 0 {
                want[b] = cand[b * k as usize];
            }
            let mut prev = tokens[b];
            for row in b + first as usize..e {
                let mut best = (0usize, f32::NEG_INFINITY);
                for c in 0..k as usize {
                    let id = cand[row * k as usize + c];
                    let mut s = 0f64;
                    if (0..vocab as i32).contains(&prev) && (0..vocab as i32).contains(&id) {
                        for d in 0..rank as usize {
                            let a = f64::from(pred[prev as usize * rank as usize + d]);
                            let bb = f64::from(succ[id as usize * rank as usize + d]);
                            let h = if with_hp {
                                f64::from(hp[row * rank as usize + d])
                            } else {
                                1.0
                            };
                            s += a * h * bb;
                        }
                    }
                    let score = unary[row * k as usize + c] + s as f32;
                    if c == 0 || score > best.1 {
                        best = (c, score);
                    }
                }
                prev = cand[row * k as usize + best.0];
                want[row] = prev;
            }
        }
        assert_eq!(got, want, "selector walk first={first} hp={with_hp}");
    }
}

// ------------------------------------------------------- score capture --

#[test]
fn score_capture_averages_the_observed_rows_mass() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (hq, hk, d, ps) = (4u32, 2u32, 128u32, 16u32);
    let fresh = [5u32, 1, 2];
    let prefix = [20u32, 33, 0];
    let kv_lens: Vec<u32> = fresh.iter().zip(&prefix).map(|(f, p)| f + p).collect();
    let Cache {
        pool,
        keys,
        page_indptr,
        ..
    } = paged_cache(&gpu, hk as usize, d as usize, ps as usize, &kv_lens, 40);
    let qo = indptr_of(&fresh);
    let total = *qo.last().unwrap() as u32;
    let qs = bf16s((total * hq * d) as usize, 17, 1.0);
    let q = RaggedTensor {
        data: gpu.tensor(&qs, total, hq * d, Dtype::Bf16),
        indptr: gpu.column(&qo),
    };
    let device = plan::Device::probe(&ctx).unwrap();
    let int = gpu.zeros(1, 4 << 20, Dtype::F32);
    let float = gpu.zeros(1, 4 << 20, Dtype::F32);
    let n = fresh.len() as u32;
    let built = plan::plan_prefill(
        &qo,
        &page_indptr,
        &kv_lens.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        total,
        plan::Shape {
            num_requests: n,
            lane_offset: 0,
            num_q_heads: hq,
            num_kv_heads: hk,
            head_dim: d,
            page_size: ps,
            hnd_layout: false,
        },
        plan::Live {
            requests: n,
            lane_offset: 0,
            row_offset: 0,
            rows: total,
        },
        None,
        true,
        false,
        None,
        &device,
        plan::Toggles::default(),
        plan::Workspace {
            int_ptr: int.ptr,
            int_bytes: 16 << 20,
            float_ptr: float.ptr,
            float_bytes: 16 << 20,
        },
    )
    .unwrap();
    let (observe, lane_offset, plane_stride, plane, kv_max) = (3u32, 1u32, hq + 2, 1u32, 40u32);
    let slab_rows = (lane_offset + n) * plane_stride;
    let mut scores = gpu.tensor(
        &vec![9.0; (slab_rows * kv_max) as usize],
        slab_rows,
        kv_max,
        Dtype::F32,
    );
    let scale = 1.0 / (d as f32).sqrt();
    kernels_rocm::attn_score::capture(
        &ctx,
        q,
        &built,
        &pool,
        None,
        d,
        hk,
        scale,
        observe,
        lane_offset,
        plane_stride,
        plane,
        kv_max,
        &mut scores,
    )
    .unwrap();
    let got = gpu.read(&scores);
    let (d_, g) = (d as usize, (hq / hk) as usize);
    for r in 0..n as usize {
        let kv = kv_lens[r] as usize;
        let rows = (observe as usize).min(fresh[r] as usize);
        for h in 0..hq as usize {
            let mut want = vec![0f64; kv_max as usize];
            for w in 0..rows {
                let t = qo[r + 1] as usize - rows + w;
                let limit = (kv - rows + w + 1).min(kv);
                let qrow = &qs[(t * hq as usize + h) * d_..(t * hq as usize + h + 1) * d_];
                let s: Vec<f64> = (0..limit)
                    .map(|j| {
                        let at = (j * hk as usize + h / g) * d_;
                        (0..d_)
                            .map(|i| f64::from(qrow[i]) * f64::from(keys[r][at + i]))
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::MIN, f64::max);
                let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
                for (j, x) in s.iter().enumerate() {
                    if j < kv_max as usize {
                        want[j] += (x - m).exp() / z / rows as f64;
                    }
                }
            }
            let at = (((lane_offset as usize + r) * plane_stride as usize) + plane as usize + h)
                * kv_max as usize;
            let want: Vec<f32> = want.iter().map(|&x| x as f32).collect();
            close_abs(
                &format!("score req {r} head {h}"),
                &got[at..at + kv_max as usize],
                &want,
                2e-5,
            );
        }
    }
    // The planes this capture does not own keep what they held.
    assert!(
        got[..(lane_offset * plane_stride * kv_max) as usize]
            .iter()
            .all(|&x| x == 9.0)
    );
}

// ------------------------------------------------------------- dynconv --

#[test]
fn block_dyn_conv_convolves_inside_each_request() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (channels, group, taps) = (300usize, 4usize, 3usize);
    let groups = channels / group;
    let lens = [5u32, 0, 7, 1];
    let indptr = indptr_of(&lens);
    let rows = *indptr.last().unwrap() as usize;
    let xs = bf16s(rows * channels, 18, 1.0);
    let cs = bf16s(rows * 2 * taps * groups, 19, 0.3);
    let bs = bf16s(2 * taps * channels, 20, 1.0);
    for side in 0..2usize {
        let x = RaggedTensor {
            data: gpu.tensor(&xs, rows as u32, channels as u32, Dtype::Bf16),
            indptr: gpu.column(&indptr),
        };
        let coeff = gpu.tensor(&cs, rows as u32, (2 * taps * groups) as u32, Dtype::Bf16);
        let base = gpu.tensor(&bs, (2 * taps) as u32, channels as u32, Dtype::Bf16);
        let mut y = gpu.zeros(rows as u32, channels as u32, Dtype::Bf16);
        kernels_rocm::attn::dynconv::block_dyn_conv(
            &ctx,
            x,
            coeff,
            base,
            side as u32,
            taps as u32,
            group as u32,
            &mut y,
        )
        .unwrap();
        let mut want = vec![0f32; rows * channels];
        for r in 0..lens.len() {
            let b = indptr[r] as usize;
            for t in 0..lens[r] as usize {
                let row = b + t;
                for c in 0..channels {
                    let mut acc = 0f32;
                    for k in 0..taps.min(t + 1) {
                        let at = side * taps + k;
                        let coef = bs[at * channels + c]
                            + cs[row * 2 * taps * groups + at * groups + c / group];
                        acc += coef * xs[(row - k) * channels + c];
                    }
                    want[row * channels + c] = acc;
                }
            }
        }
        close_bf16(&format!("dyn conv side {side}"), &gpu.read(&y), &want, 1.0);
    }
}

// ---------------------------------------------------------------- ple --

const MULTS: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xc2b2_ae3d_27d4_eb4f,
    0x1656_67b1_9e37_79f9,
];
const PRIMES: [u64; 4] = [1_000_003, 999_983, 1_000_033, 999_979];
const OFFSETS: [u64; 4] = [0, 1_000_003, 2_000_000, 3_000_000];
const EOS: i32 = 2;

/// One token's ids from its window (token, then predecessors).
fn ple_row(mut window: Vec<i32>, map: Option<&[i64]>) -> Vec<i32> {
    let mut crossed = false;
    for w in window.iter_mut().skip(1) {
        if crossed {
            *w = EOS;
        }
        if *w == EOS {
            crossed = true;
        }
    }
    if let Some(map) = map {
        for w in &mut window {
            *w = map[*w as usize] as i32;
        }
    }
    let mut out = Vec::new();
    for order in 2..=window.len() {
        let mut mixed = (window[0] as u64).wrapping_mul(MULTS[0]);
        for p in 1..order {
            mixed ^= (window[p] as u64).wrapping_mul(MULTS[p]);
        }
        for k in 0..2 {
            let head = (order - 2) * 2 + k;
            out.push((mixed % PRIMES[head] + OFFSETS[head]) as i32);
        }
    }
    out
}

fn cell_id(cell: i32) -> i32 {
    if cell == 0 { EOS } else { cell - 1 }
}

#[test]
fn ple_ngram_ids_hash_windows_across_the_state() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let stride = 4usize;
    // Slot states (span 2, + 1 encoded, oldest first); slot 1 holds an eos.
    let state0: Vec<i32> = vec![11, 21, 0, 0, 3, 41, 0, 0, 0, 0, 0, 0, 17, 9, 0, 0];
    let map: Vec<i64> = (0..64).map(|i| (i * 5 + 1) % 61).collect();

    for with_map in [false, true] {
        let mp = with_map.then_some(&map[..]);
        let map_t = with_map.then(|| gpu.upload(&map, 64, 1, Dtype::I64));

        // Decode: one row per slot; slot −1 skips.
        let ids = [7i32, 2, 30, 12];
        let slots = [0i32, 1, -1, 3];
        let slab = gpu.upload(&state0, 4, stride as u32, Dtype::I32);
        let pool = RecurrentPool {
            slab,
            slot_ids: gpu.column(&slots),
            slot_stride_elems: stride as i64,
            ..STATE
        };
        let mut out = gpu.upload(&[-1i32; 4 * 4], 4, 4, Dtype::I32);
        kernels_rocm::attn_ple::ngram_ids(
            &ctx,
            gpu.column(&ids),
            &pool,
            EOS as u32,
            &MULTS,
            &PRIMES,
            &OFFSETS,
            2,
            map_t,
            &mut out,
        )
        .unwrap();
        let (got, state) = (gpu.read_as::<i32>(&out), gpu.read_as::<i32>(&slab));
        let mut want_state = state0.clone();
        for r in 0..4 {
            let want: Vec<i32> = if slots[r] < 0 {
                vec![-1; 4]
            } else {
                let s = &state0[slots[r] as usize * stride..];
                let ws = &mut want_state[slots[r] as usize * stride..];
                (ws[0], ws[1]) = (s[1], ids[r] + 1);
                ple_row(vec![ids[r], cell_id(s[1]), cell_id(s[0])], mp)
            };
            assert_eq!(
                &got[r * 4..(r + 1) * 4],
                &want[..],
                "ple decode row {r} map={with_map}"
            );
        }
        assert_eq!(state, want_state, "ple decode state map={with_map}");

        // Chunked: ragged requests with a begin offset and a commit cap.
        let lens = [5u32, 1, 0, 3];
        let qo = indptr_of(&lens);
        let rows = *qo.last().unwrap() as usize;
        let tok: Vec<i32> = (0..rows).map(|i| ((i * 13 + 5) % 60) as i32).collect();
        let begin = [1i32, 0, 0, 0];
        let commit = [3i32, 1, 0, 9];
        let mask = [1u8, 0, 1, 1];
        let slab = gpu.upload(&state0, 4, stride as u32, Dtype::I32);
        let pool = RecurrentPool {
            slab,
            slot_ids: gpu.column(&[0, 1, 2, 3]),
            slot_stride_elems: stride as i64,
            write_state_mask: gpu.upload(&mask, 4, 1, Dtype::U8),
            commit_len: gpu.column(&commit),
            begin_at: gpu.column(&begin),
            ..STATE
        };
        let mut out = gpu.upload(&vec![-1i32; rows * 4], rows as u32, 4, Dtype::I32);
        let x = RaggedTensor {
            data: gpu.column(&tok),
            indptr: gpu.column(&qo),
        };
        kernels_rocm::attn_ple::ngram_ids_chunked(
            &ctx, x, &pool, EOS as u32, &MULTS, &PRIMES, &OFFSETS, 2, map_t, &mut out,
        )
        .unwrap();
        let (got, state) = (gpu.read_as::<i32>(&out), gpu.read_as::<i32>(&slab));
        let mut want = vec![-1i32; rows * 4];
        let mut want_state = state0.clone();
        for r in 0..lens.len() {
            let mut t0 = qo[r] as usize;
            let mut n = lens[r] as usize;
            let b = (begin[r] as usize).min(n);
            t0 += b;
            n -= b;
            n = n.min(commit[r] as usize);
            if n == 0 {
                continue;
            }
            let s = &state0[r * stride..(r + 1) * stride];
            for t in 0..n {
                let pred = |p: usize| {
                    if t >= p {
                        tok[t0 + t - p]
                    } else {
                        cell_id(s[2 - (p - t)])
                    }
                };
                let ids = ple_row(vec![tok[t0 + t], pred(1), pred(2)], mp);
                want[(t0 + t) * 4..(t0 + t + 1) * 4].copy_from_slice(&ids);
            }
            if mask[r] != 0 {
                let next: Vec<i32> = (0..2)
                    .map(|p| {
                        let src = n as i64 - 2 + p as i64;
                        if src >= 0 {
                            tok[t0 + src as usize] + 1
                        } else {
                            s[p + n]
                        }
                    })
                    .collect();
                want_state[r * stride..r * stride + 2].copy_from_slice(&next);
            }
        }
        assert_eq!(got, want, "ple chunked ids map={with_map}");
        assert_eq!(state, want_state, "ple chunked state map={with_map}");
    }
}

// --------------------------------------------------------- fused qkv --

#[test]
fn fused_qkv_norms_ropes_and_writes_the_kv_slot() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // 64/128/256/512 run the wavefront entry, 96 the block entry.
    for (d, rotary, hnd) in [
        (128u32, 128u32, false),
        (64, 32, true),
        (256, 256, false),
        (512, 128, true),
        (96, 64, true),
    ] {
        let (hq, hk, ps, rows) = (4u32, 2u32, 8u32, 5u32);
        let width = (hq + 2 * hk) * d;
        let packed_s = bf16s((rows * width) as usize, 22, 2.0);
        let qw = bf16s(d as usize, 23, 1.0);
        let kw = bf16s(d as usize, 24, 1.0);
        let positions: Vec<i32> = (0..rows as i32).map(|r| r * 977 + 3).collect();
        let pages = 4usize;
        let slots = pages * ps as usize;
        let row = (hk * d) as usize;
        let w_page = [3i32, 0, 2, 2, 1];
        let w_off = [7i32, 0, 1, 2, 5];
        let (theta, eps) = (1e6f32, 1e-6f32);

        let run = |win: Option<[i32; 4]>| {
            let pool = KvPool {
                keys: gpu.zeros(slots as u32, row as u32, Dtype::Bf16),
                values: gpu.zeros(slots as u32, row as u32, Dtype::Bf16),
                page_size: ps as i32,
                seq_stride: if hnd { i64::from(d) } else { row as i64 },
                head_stride: if hnd { i64::from(ps * d) } else { i64::from(d) },
                layout: i32::from(hnd),
                ..KV
            };
            let mut q = gpu.zeros(rows, hq * d, Dtype::Bf16);
            if let Some(w) = win {
                ctx.arm_stage(gpu.ints(&w).ptr);
            }
            kernels_rocm::custom::qkv_fused_qknorm_rope_vnorm_write(
                &ctx,
                gpu.tensor(&packed_s, rows, width, Dtype::Bf16),
                Some(gpu.column(&positions)),
                Some(gpu.tensor(&qw, 1, d, Dtype::Bf16)),
                Some(gpu.tensor(&kw, 1, d, Dtype::Bf16)),
                true,
                eps,
                &pool,
                gpu.column(&w_page),
                gpu.column(&w_off),
                hk,
                d,
                theta,
                rotary,
                d / 2,
                true,
                1.0,
                [d; 2],
                &mut q,
            )
            .unwrap();
            ctx.disarm_stage();
            (gpu.read(&q), gpu.read(&pool.keys), gpu.read(&pool.values))
        };

        let d_ = d as usize;
        let normed = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
            let ss: f32 = x.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / d as f32 + eps).sqrt();
            x.iter()
                .enumerate()
                .map(|(i, &v)| w.map_or(v * inv, |w| v * inv * w[i]))
                .collect()
        };
        let roped = |x: Vec<f32>, pos: i32| -> Vec<f32> {
            let half = d_ / 2;
            let mut y = x.clone();
            for i in 0..half {
                let (c, s) = if i < rotary as usize / 2 {
                    let f = (f64::from(theta)).powf(-2.0 * i as f64 / f64::from(d));
                    let a = f64::from(pos) * f;
                    (a.cos() as f32, a.sin() as f32)
                } else {
                    (1.0, 0.0)
                };
                y[i] = x[i] * c - x[i + half] * s;
                y[i + half] = x[i + half] * c + x[i] * s;
            }
            y
        };
        let kv_at = |page: usize, slot: usize, h: usize| {
            if hnd {
                ((page * hk as usize + h) * ps as usize + slot) * d_
            } else {
                (page * ps as usize + slot) * row + h * d_
            }
        };

        for win in [None, Some([3, 1, 0, 0])] {
            let (q, k, v) = run(win);
            for r in 0..rows as usize {
                let live = win.is_none() || (1..4).contains(&r);
                let src = &packed_s[r * width as usize..(r + 1) * width as usize];
                for h in 0..hq as usize {
                    let have = &q[(r * hq as usize + h) * d_..(r * hq as usize + h + 1) * d_];
                    let tag = format!("fused d{d} hnd={hnd} win={win:?} row {r} q head {h}");
                    if live {
                        let want =
                            roped(normed(&src[h * d_..(h + 1) * d_], Some(&qw)), positions[r]);
                        close_abs(&tag, have, &want, 2e-2);
                    } else {
                        assert!(
                            have.iter().all(|&x| x == 0.0),
                            "{tag}: a dead row is untouched"
                        );
                    }
                }
                if !live {
                    continue;
                }
                for h in 0..hk as usize {
                    let at = kv_at(w_page[r] as usize, w_off[r] as usize, h);
                    let ks = &src[(hq as usize + h) * d_..(hq as usize + h + 1) * d_];
                    let vs = &src[((hq + hk) as usize + h) * d_..((hq + hk) as usize + h + 1) * d_];
                    let tag = format!("fused d{d} hnd={hnd} win={win:?} row {r} kv head {h}");
                    close_abs(
                        &format!("{tag} k"),
                        &k[at..at + d_],
                        &roped(normed(ks, Some(&kw)), positions[r]),
                        2e-2,
                    );
                    close_bf16(&format!("{tag} v"), &v[at..at + d_], &normed(vs, None), 1.0);
                }
            }
        }
    }
}

// ------------------------------------------------------------ rel bias --

/// Inkling's relative-bias attention against the host: the bias added at
/// each key's distance inside `extent`, the logits grown by
/// `1 + α·ln((p + 1) / floor)` past the floor, a sliding window or none, over
/// decode rows and prefill rows that share requests.
#[allow(clippy::too_many_arguments)]
fn check_rel(
    gpu: &Gpu,
    hq: u32,
    hk: u32,
    d: u32,
    fresh: &[u32],
    prefix: &[u32],
    extent: u32,
    window: Option<u32>,
    log: (u32, f32),
    seed: u64,
) {
    let ctx = gpu.ctx();
    let ps = 16;
    let kv_lens: Vec<u32> = fresh.iter().zip(prefix).map(|(f, p)| f + p).collect();
    let Cache {
        pool, keys, vals, ..
    } = paged_cache(gpu, hk as usize, d as usize, ps, &kv_lens, seed);
    let qo = indptr_of(fresh);
    let total = *qo.last().unwrap() as usize;
    let qs = bf16s(total * (hq * d) as usize, seed + 7, 1.0);
    let bias = values(total * (hq * extent) as usize, seed + 9, 2.0);
    let q = RaggedTensor {
        data: gpu.tensor(&qs, total as u32, hq * d, Dtype::Bf16),
        indptr: gpu.column(&qo),
    };
    let scale = 1.0 / (d as f32).sqrt();
    let mut o = gpu.zeros(total as u32, hq * d, Dtype::Bf16);
    let rel = attn::RelBias {
        bias: gpu.tensor(&bias, total as u32, hq * extent, Dtype::F32),
        extent,
        log_floor: log.0,
        log_alpha: log.1,
    };
    let window_left = window.map_or(-1, |w| w as i32 - 1);
    attn::rel::walk(
        &ctx,
        "attention.prefill_rel",
        q,
        &pool,
        rel,
        window_left,
        d,
        hk,
        scale,
        &mut o,
    )
    .unwrap();
    let got = gpu.read(&o);
    let (du, g) = (d as usize, (hq / hk) as usize);
    for r in 0..fresh.len() {
        for i in 0..fresh[r] as usize {
            let t = qo[r] as usize + i;
            let p = prefix[r] as usize + i;
            let lo = window.map_or(0, |w| (p + 1).saturating_sub(w as usize));
            let grow = if log.1 != 0.0 && log.0 != 0 && p + 1 > log.0 as usize {
                1.0 + f64::from(log.1) * ((p + 1) as f64 / f64::from(log.0)).ln()
            } else {
                1.0
            };
            for h in 0..hq as usize {
                let kh = h / g;
                let at = |j: usize| (j * hk as usize + kh) * du;
                let qh = &qs[(t * hq as usize + h) * du..(t * hq as usize + h + 1) * du];
                let s: Vec<f64> = (lo..=p)
                    .map(|j| {
                        let dot: f64 = (0..du)
                            .map(|e| f64::from(qh[e]) * f64::from(keys[r][at(j) + e]))
                            .sum();
                        let mut s = dot * f64::from(scale);
                        if p - j < extent as usize {
                            s += f64::from(bias[(t * hq as usize + h) * extent as usize + p - j]);
                        }
                        s * grow
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                let want: Vec<f32> = (0..du)
                    .map(|c| {
                        ((lo..=p)
                            .zip(&e)
                            .map(|(j, w)| w * f64::from(vals[r][at(j) + c]))
                            .sum::<f64>()
                            / z) as f32
                    })
                    .collect();
                close_abs(
                    &format!("rel h{hq}/{hk} d{d} w{window:?} row {t} head {h}"),
                    &got[(t * hq as usize + h) * du..(t * hq as usize + h + 1) * du],
                    &want,
                    1e-2,
                );
            }
        }
    }
}

#[test]
fn relative_bias_attention_answers_the_host() {
    let Some(gpu) = Gpu::open() else { return };
    // Inkling-Small's global layer: 32 heads over 8, 128 wide, the log
    // growth past a short floor; decode rows and a prefill.
    check_rel(
        &gpu,
        32,
        8,
        128,
        &[1, 1, 1],
        &[0, 70, 300],
        16,
        None,
        (64, 0.1),
        201,
    );
    check_rel(
        &gpu,
        32,
        8,
        128,
        &[90, 3],
        &[40, 0],
        16,
        None,
        (64, 0.1),
        202,
    );
    // Its local layers: a sliding window, no growth.
    check_rel(
        &gpu,
        32,
        8,
        128,
        &[1, 70],
        &[300, 100],
        16,
        Some(128),
        (0, 0.0),
        203,
    );
    // A narrow head and a long extent.
    check_rel(
        &gpu,
        6,
        2,
        64,
        &[5, 1],
        &[20, 200],
        300,
        Some(50),
        (100, 0.3),
        204,
    );
}
