//! The sparse-attention indexer on the first wgpu adapter, beside the hand-written
//! kernels: the index keys' layer norm and rope and the query's rope against
//! an f64 reference; the top-k scores against an f64 reference and its
//! selection exactly against the bisection's host replica, with and without
//! head weights, at ratio 1 and 2, a row short of its budget included.

use kernels_wgpu::{KvPool, Tensor, attn::index};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, close, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn sp(p: &KvPool) -> kernels_slang::KvPool {
    kernels_slang::KvPool {
        keys: st(p.keys),
        values: st(p.values),
        page_indices: st(p.page_indices),
        page_indptr: st(p.page_indptr),
        page_size: p.page_size,
        seq_stride: p.seq_stride,
        head_stride: p.head_stride,
    }
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

/// Interleaved pairs of the first `rope` columns of `row` turned by `pos`.
fn turn(row: &mut [f64], rope: usize, pos: i32, theta: f64) {
    for i in 0..rope / 2 {
        let ang = f64::from(pos) * theta.powf(-2.0 * i as f64 / rope as f64);
        let (a, b) = (row[2 * i], row[2 * i + 1]);
        row[2 * i] = a * ang.cos() - b * ang.sin();
        row[2 * i + 1] = b * ang.cos() + a * ang.sin();
    }
}

#[test]
fn the_indexer_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let b = |buf: u32, r: usize, w: usize| Tensor::new(buf, r as u32, w as u32, Dtype::Bf16);
    let i32t = |buf: u32, r: usize| Tensor::new(buf, r as u32, 1, Dtype::I32);
    let (d, rope, theta, eps) = (128usize, 64usize, 10_000.0f32, 1e-6f32);
    let rows = 3usize;
    let positions = [0i32, 9, 30];
    let pos = gpu.up_i32(&positions);

    // the key norm and rope
    let kv = noise(1, rows * d)
        .iter()
        .map(|v| v + 0.5)
        .collect::<Vec<_>>();
    let kr = f64s(&gpu.up(&kv).1);
    let (w, wr) = gpu.up(&noise(2, d));
    let (bb, br) = gpu.up(&noise(3, d));
    let (wr, br) = (f64s(&wr), f64s(&br));
    let mut want = Vec::new();
    for r in 0..rows {
        let x = &kr[r * d..][..d];
        let mean = x.iter().sum::<f64>() / d as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        // The normed row is rounded to bf16 before it turns.
        let mut y: Vec<f64> = (0..d)
            .map(|c| f64::from(half_round(((x[c] - mean) * inv * wr[c] + br[c]) as f32)))
            .collect();
        turn(&mut y, rope, positions[r], f64::from(theta));
        want.extend(y);
    }
    both(
        &mut gpu,
        "index key norm rope",
        &|g| vec![(g.up(&kv).0, rows * d, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                index::layernorm_rope(
                    c,
                    b(o[0].0, rows, d),
                    i32t(pos, rows),
                    b(w, 1, d),
                    b(bb, 1, d),
                    eps,
                    rope as u32,
                    theta,
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::index::layernorm_rope(
                    c,
                    st(b(o[0].0, rows, d)),
                    st(i32t(pos, rows)),
                    st(b(w, 1, d)),
                    st(b(bb, 1, d)),
                    eps,
                    rope as u32,
                    theta,
                )
            })
        },
    );

    // the query rope, 4 heads
    let heads = 4usize;
    let qv = noise(4, rows * heads * d);
    let qr = f64s(&gpu.up(&qv).1);
    let mut want = qr.clone();
    for r in 0..rows {
        for h in 0..heads {
            turn(
                &mut want[(r * heads + h) * d..][..d],
                rope,
                positions[r],
                f64::from(theta),
            );
        }
    }
    both(
        &mut gpu,
        "index q rope",
        &|g| vec![(g.up(&qv).0, rows * heads * d, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                index::rope(
                    c,
                    b(o[0].0, rows, heads * d),
                    i32t(pos, rows),
                    heads as u32,
                    d as u32,
                    rope as u32,
                    theta,
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::index::rope(
                    c,
                    st(b(o[0].0, rows, heads * d)),
                    st(i32t(pos, rows)),
                    heads as u32,
                    d as u32,
                    rope as u32,
                    theta,
                )
            })
        },
    );

    // The index cache: request 0 (rows 0, 1) reads pages [3, 1, 0, 2], request
    // 1 (row 2) pages [4..12) of 4 slots; 12 pages, 48 slots.
    let page = 4usize;
    let page_indices: Vec<u32> = [3u32, 1, 0, 2].into_iter().chain(4..12).collect();
    let indptr = [0u32, 4, 12];
    let (pi, pp) = (gpu.up_u32(&page_indices), gpu.up_u32(&indptr));
    let u = |buf: u32, r: usize| Tensor::new(buf, r as u32, 1, Dtype::U32);
    let slots = 12 * page;
    let (keys, keyr) = gpu.up(&noise(5, slots * d));
    let keyr = f64s(&keyr);
    let pool = |k: u32| KvPool {
        keys: b(k, slots, d),
        values: b(k, slots, d),
        page_indices: u(pi, page_indices.len()),
        page_indptr: u(pp, 3),
        page_size: page as i32,
        seq_stride: d as u64,
        head_stride: d as u64,
    };
    let owner = [0i32, 0, 1];
    let own = gpu.up_i32(&owner);
    let (q, qr) = gpu.up(&noise(6, rows * heads * d));
    let qr = f64s(&qr);
    let (hw, hwr) = gpu.up(&noise(7, rows * heads));
    let hwr = f64s(&hwr);
    let top_k = 8usize;
    let stride = 32usize;
    for (ratio, weighted) in [(1usize, false), (2, true)] {
        let mut want_scores = Vec::new();
        for r in 0..rows {
            let my = &page_indices
                [indptr[owner[r] as usize] as usize..indptr[owner[r] as usize + 1] as usize];
            let total = positions[r] as usize + 1;
            let nkeys = (total / ratio).min(stride);
            for j in 0..nkeys {
                let cell = (j + 1) * ratio - 1;
                let slot = my[cell / page] as usize * page + cell % page;
                let s: f64 = (0..heads)
                    .map(|h| {
                        let dot: f64 = (0..d)
                            .map(|c| qr[(r * heads + h) * d + c] * keyr[slot * d + c])
                            .sum();
                        dot.max(0.0) * if weighted { hwr[r * heads + h] } else { 1.0 }
                    })
                    .sum();
                want_scores.push((r, j, s));
            }
        }
        let weights = weighted.then(|| b(hw, rows, heads));
        for slang in [false, true] {
            let sc = gpu.up_f32(&vec![0.0; rows * stride]);
            let sel = gpu.up_i32(&vec![-9; rows * top_k]);
            let (sct, selt) = (
                Tensor::new(sc, rows as u32, stride as u32, Dtype::F32),
                Tensor::new(sel, rows as u32, top_k as u32, Dtype::I32),
            );
            let qt = b(q, rows, heads * d);
            if slang {
                gpu.run_slang(|c| {
                    kernels_slang::attn::index::topk(
                        c,
                        st(qt),
                        weights.map(st),
                        &sp(&pool(keys)),
                        st(i32t(pos, rows)),
                        st(i32t(own, rows)),
                        st(sct),
                        heads as u32,
                        d as u32,
                        top_k as u32,
                        ratio as u32,
                        st(selt),
                    )
                });
            } else {
                gpu.run(|c| {
                    index::topk(
                        c,
                        qt,
                        weights,
                        &pool(keys),
                        i32t(pos, rows),
                        i32t(own, rows),
                        sct,
                        heads as u32,
                        d as u32,
                        top_k as u32,
                        ratio as u32,
                        selt,
                    )
                });
            }
            let who = if slang { "slang " } else { "" };
            let got = gpu.down_f32(sc, rows * stride);
            let picked: Vec<f32> = want_scores
                .iter()
                .map(|(r, j, _)| got[r * stride + j])
                .collect();
            let want: Vec<f64> = want_scores.iter().map(|(_, _, s)| *s).collect();
            close(
                &picked,
                &want,
                &format!("{who}index scores ratio {ratio} weighted {weighted}"),
            );
            let selected: Vec<i32> = gpu
                .down_f32(sel, rows * top_k)
                .iter()
                .map(|v| v.to_bits() as i32)
                .collect();
            for r in 0..rows {
                let nkeys = ((positions[r] as usize + 1) / ratio).min(stride);
                let expect = index::bisect_select(&got[r * stride..][..nkeys], top_k);
                assert_eq!(
                    &selected[r * top_k..][..top_k],
                    &expect[..],
                    "{who}index selection ratio {ratio} row {r}"
                );
            }
            eprintln!("{who}index selection ratio {ratio} weighted {weighted}: exact");
        }
    }

    // the append, through the MLA appender's no-rope form
    let (kn, knr) = gpu.up(&noise(8, rows * d));
    let knr = f64s(&knr);
    let (wpg, wof) = ([5u32, 0, 11], [2u32, 3, 0]);
    let (wp, wo) = (gpu.up_u32(&wpg), gpu.up_u32(&wof));
    let mut want = vec![0.0; slots * d];
    for r in 0..rows {
        let slot = wpg[r] as usize * page + wof[r] as usize;
        want[slot * d..][..d].copy_from_slice(&knr[r * d..][..d]);
    }
    both(
        &mut gpu,
        "index kv append",
        &|g| vec![(g.zeros(slots * d), slots * d, false)],
        &[want],
        &|g, o| {
            g.run(|c| index::kv_append(c, b(kn, rows, d), &pool(o[0].0), u(wp, rows), u(wo, rows)))
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::index::kv_append(
                    c,
                    st(b(kn, rows, d)),
                    &sp(&pool(o[0].0)),
                    st(u(wp, rows)),
                    st(u(wo, rows)),
                )
            })
        },
    );
}

/// `v` rounded to bf16, ties to even.
fn half_round(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}
