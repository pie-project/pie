//! Multi-head latent attention on the first Vulkan device, beside the hand-written
//! kernels, against an f64 host reference: the latent prologue (latents,
//! the q split, the absorbed projections, the paged append) and the paged
//! attention over latents, causal and over a selection.

use kernels_vulkan::{KvPool, RaggedTensor, Tensor, attn::mla};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

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

const HEADS: usize = 2;
const RANK: usize = 128;
const ROPE: usize = 64;
const NOPE: usize = 32;
const VDIM: usize = 32;
const PAGE: usize = 4;

#[test]
fn mla_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let b = |buf: u32, r: usize, w: usize| Tensor::new(buf, r as u32, w as u32, Dtype::Bf16);
    let rows = 3usize;

    // latents: the normed rank beside the rope tail of each projected row.
    let width = RANK + ROPE;
    let (kv_a, ar) = gpu.up(&noise(1, rows * width));
    let (wn, wr) = gpu.up(&noise(2, RANK));
    let (ar, wr) = (f64s(&ar), f64s(&wr));
    let eps = 1e-6f32;
    let mut want_c = Vec::new();
    let mut want_pe = Vec::new();
    for r in 0..rows {
        let lat = &ar[r * width..][..RANK];
        let inv =
            1.0 / (lat.iter().map(|v| v * v).sum::<f64>() / RANK as f64 + f64::from(eps)).sqrt();
        want_c.extend(lat.iter().zip(&wr).map(|(v, w)| v * inv * w));
        want_pe.extend_from_slice(&ar[r * width + RANK..][..ROPE]);
    }
    both(
        &mut gpu,
        "mla latents",
        &|g| {
            vec![
                (g.zeros(rows * RANK), rows * RANK, false),
                (g.zeros(rows * ROPE), rows * ROPE, false),
            ]
        },
        &[want_c, want_pe],
        &|g, o| {
            g.run(|c| {
                mla::latents(
                    c,
                    b(kv_a, rows, width),
                    b(wn, 1, RANK),
                    eps,
                    RANK as u32,
                    b(o[0].0, rows, RANK),
                    b(o[1].0, rows, ROPE),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::latents(
                    c,
                    st(b(kv_a, rows, width)),
                    st(b(wn, 1, RANK)),
                    eps,
                    RANK as u32,
                    st(b(o[0].0, rows, RANK)),
                    st(b(o[1].0, rows, ROPE)),
                )
            })
        },
    );

    // the q split: each head's nope | rope cut apart.
    let per = NOPE + ROPE;
    let (q_b, qbr) = gpu.up(&noise(3, rows * HEADS * per));
    let qbr = f64s(&qbr);
    let want_n: Vec<f64> = (0..rows * HEADS * NOPE)
        .map(|i| qbr[(i / NOPE) * per + i % NOPE])
        .collect();
    let want_p: Vec<f64> = (0..rows * HEADS * ROPE)
        .map(|i| qbr[(i / ROPE) * per + NOPE + i % ROPE])
        .collect();
    both(
        &mut gpu,
        "mla split q",
        &|g| {
            vec![
                (g.zeros(rows * HEADS * NOPE), rows * HEADS * NOPE, false),
                (g.zeros(rows * HEADS * ROPE), rows * HEADS * ROPE, false),
            ]
        },
        &[want_n, want_p],
        &|g, o| {
            g.run(|c| {
                mla::split_q_b(
                    c,
                    b(q_b, rows, HEADS * per),
                    HEADS as u32,
                    NOPE as u32,
                    ROPE as u32,
                    b(o[0].0, rows, HEADS * NOPE),
                    b(o[1].0, rows, HEADS * ROPE),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::split_q_b(
                    c,
                    st(b(q_b, rows, HEADS * per)),
                    HEADS as u32,
                    NOPE as u32,
                    ROPE as u32,
                    st(b(o[0].0, rows, HEADS * NOPE)),
                    st(b(o[1].0, rows, HEADS * ROPE)),
                )
            })
        },
    );

    // the absorbed projections, through kv_b laid [head][nope + v][rank].
    let (kv_b, kbr) = gpu.up(&noise(4, HEADS * (NOPE + VDIM) * RANK)
        .iter()
        .map(|v| v * 0.2)
        .collect::<Vec<_>>());
    let (qn, qnr) = gpu.up(&noise(5, rows * HEADS * NOPE));
    let (kbr, qnr) = (f64s(&kbr), f64s(&qnr));
    let wb = |h: usize, j: usize, i: usize| kbr[(h * (NOPE + VDIM) + j) * RANK + i];
    let want: Vec<f64> = (0..rows * HEADS * RANK)
        .map(|x| {
            let (t, h, i) = (x / (HEADS * RANK), (x / RANK) % HEADS, x % RANK);
            (0..NOPE)
                .map(|j| qnr[(t * HEADS + h) * NOPE + j] * wb(h, j, i))
                .sum()
        })
        .collect();
    both(
        &mut gpu,
        "mla absorb q",
        &|g| vec![(g.zeros(rows * HEADS * RANK), rows * HEADS * RANK, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                mla::absorb_q(
                    c,
                    b(qn, rows, HEADS * NOPE),
                    b(kv_b, 1, 1),
                    HEADS as u32,
                    RANK as u32,
                    NOPE as u32,
                    VDIM as u32,
                    b(o[0].0, rows, HEADS * RANK),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::absorb_q(
                    c,
                    st(b(qn, rows, HEADS * NOPE)),
                    st(b(kv_b, 1, 1)),
                    HEADS as u32,
                    RANK as u32,
                    NOPE as u32,
                    VDIM as u32,
                    st(b(o[0].0, rows, HEADS * RANK)),
                )
            })
        },
    );
    let (lat, latr) = gpu.up(&noise(6, rows * HEADS * RANK));
    let latr = f64s(&latr);
    let want: Vec<f64> = (0..rows * HEADS * VDIM)
        .map(|x| {
            let (t, h, j) = (x / (HEADS * VDIM), (x / VDIM) % HEADS, x % VDIM);
            (0..RANK)
                .map(|i| latr[(t * HEADS + h) * RANK + i] * wb(h, NOPE + j, i))
                .sum()
        })
        .collect();
    both(
        &mut gpu,
        "mla absorb out",
        &|g| vec![(g.zeros(rows * HEADS * VDIM), rows * HEADS * VDIM, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                mla::absorb_out(
                    c,
                    b(lat, rows, HEADS * RANK),
                    b(kv_b, 1, 1),
                    HEADS as u32,
                    RANK as u32,
                    VDIM as u32,
                    NOPE as u32,
                    b(o[0].0, rows, HEADS * VDIM),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::absorb_out(
                    c,
                    st(b(lat, rows, HEADS * RANK)),
                    st(b(kv_b, 1, 1)),
                    HEADS as u32,
                    RANK as u32,
                    VDIM as u32,
                    NOPE as u32,
                    st(b(o[0].0, rows, HEADS * VDIM)),
                )
            })
        },
    );

    // The paged cache: request 0 reads pages [2, 0], request 1 page [1].
    let slots = 3 * PAGE;
    let pages = [2u32, 0, 1];
    let indptr = [0u32, 2, 3];
    let (pi, pp) = (gpu.up_u32(&pages), gpu.up_u32(&indptr));
    let u = |buf: u32, r: usize| Tensor::new(buf, r as u32, 1, Dtype::U32);
    let pool = |ckv: u32, kpe: u32| KvPool {
        keys: b(ckv, slots, RANK),
        values: b(kpe, slots, ROPE),
        page_indices: u(pi, 3),
        page_indptr: u(pp, 3),
        page_size: PAGE as i32,
        seq_stride: RANK as u64,
        head_stride: RANK as u64,
    };

    // the append: three latent rows into slots (page 2, 3), (0, 1), (1, 0).
    let (kc, kcr) = gpu.up(&noise(7, rows * RANK));
    let (kp, kpr) = gpu.up(&noise(8, rows * ROPE));
    let (kcr, kpr) = (f64s(&kcr), f64s(&kpr));
    let (wpg, wof) = ([2u32, 0, 1], [3u32, 1, 0]);
    let (wp, wo) = (gpu.up_u32(&wpg), gpu.up_u32(&wof));
    let mut want_ckv = vec![0.0; slots * RANK];
    let mut want_kpe = vec![0.0; slots * ROPE];
    for r in 0..rows {
        let slot = wpg[r] as usize * PAGE + wof[r] as usize;
        want_ckv[slot * RANK..][..RANK].copy_from_slice(&kcr[r * RANK..][..RANK]);
        want_kpe[slot * ROPE..][..ROPE].copy_from_slice(&kpr[r * ROPE..][..ROPE]);
    }
    both(
        &mut gpu,
        "mla kv append",
        &|g| {
            vec![
                (g.zeros(slots * RANK), slots * RANK, false),
                (g.zeros(slots * ROPE), slots * ROPE, false),
            ]
        },
        &[want_ckv, want_kpe],
        &|g, o| {
            g.run(|c| {
                mla::kv_append(
                    c,
                    b(kc, rows, RANK),
                    b(kp, rows, ROPE),
                    &pool(o[0].0, o[1].0),
                    u(wp, rows),
                    u(wo, rows),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::kv_append(
                    c,
                    st(b(kc, rows, RANK)),
                    st(b(kp, rows, ROPE)),
                    &sp(&pool(o[0].0, o[1].0)),
                    st(u(wp, rows)),
                    st(u(wo, rows)),
                )
            })
        },
    );

    // Attention: rows 0, 1 belong to request 0 at positions 5, 7; row 2 to
    // request 1 at position 2.
    let (ckv, ckvr) = gpu.up(&noise(9, slots * RANK));
    let (kpe, kper) = gpu.up(&noise(10, slots * ROPE));
    let (qa, qar) = gpu.up(&noise(11, rows * HEADS * RANK));
    let (qpe, qper) = gpu.up(&noise(12, rows * HEADS * ROPE));
    let (ckvr, kper, qar, qper) = (f64s(&ckvr), f64s(&kper), f64s(&qar), f64s(&qper));
    let positions = [5i32, 7, 2];
    let owner = [0i32, 0, 1];
    let (pos, own) = (gpu.up_i32(&positions), gpu.up_i32(&owner));
    let i32t = |buf: u32, r: usize, w: usize| Tensor::new(buf, r as u32, w as u32, Dtype::I32);
    let scale = 0.08f64;
    let reading = |keys: &dyn Fn(usize) -> Vec<usize>| -> Vec<f64> {
        let mut out = vec![0.0; rows * HEADS * RANK];
        for row in 0..rows {
            let req = owner[row] as usize;
            let my = &pages[indptr[req] as usize..indptr[req + 1] as usize];
            let ks = keys(row);
            for h in 0..HEADS {
                let qn = &qar[(row * HEADS + h) * RANK..][..RANK];
                let qp = &qper[(row * HEADS + h) * ROPE..][..ROPE];
                let slot = |j: usize| my[j / PAGE] as usize * PAGE + j % PAGE;
                let s: Vec<f64> = ks
                    .iter()
                    .map(|&j| {
                        let sl = slot(j);
                        (qn.iter()
                            .zip(&ckvr[sl * RANK..][..RANK])
                            .map(|(a, c)| a * c)
                            .sum::<f64>()
                            + qp.iter()
                                .zip(&kper[sl * ROPE..][..ROPE])
                                .map(|(a, c)| a * c)
                                .sum::<f64>())
                            * scale
                    })
                    .collect();
                let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = s.iter().map(|v| (v - top).exp()).collect();
                let sum: f64 = w.iter().sum();
                for d in 0..RANK {
                    out[(row * HEADS + h) * RANK + d] = ks
                        .iter()
                        .zip(&w)
                        .map(|(&j, w)| w * ckvr[slot(j) * RANK + d])
                        .sum::<f64>()
                        / sum;
                }
            }
        }
        out
    };
    let causal = reading(&|row| (0..=positions[row] as usize).collect());
    let sel_v = [7i32, 2, -1, 9, 0, 5, 9, 1, 2, 0, 3, 1];
    let sel = gpu.up_i32(&sel_v);
    let selected = reading(&|row| {
        sel_v[row * 4..][..4]
            .iter()
            .filter(|&&j| j >= 0 && j <= positions[row])
            .map(|&j| j as usize)
            .collect()
    });
    let n = rows * HEADS * RANK;
    let (q_t, qpe_t, o_t) = (
        b(qa, rows, HEADS * RANK),
        b(qpe, rows, HEADS * ROPE),
        |buf: u32| b(buf, rows, HEADS * RANK),
    );
    let (pos_t, own_t, sel_t) = (i32t(pos, rows, 1), i32t(own, rows, 1), i32t(sel, rows, 4));
    let qp_i = gpu.up_i32(&[0, 2, 3]);
    let ragged = RaggedTensor {
        data: q_t,
        indptr: i32t(qp_i, 3, 1),
    };
    let sragged = kernels_slang::RaggedTensor {
        data: st(q_t),
        indptr: st(i32t(qp_i, 3, 1)),
    };
    let cache = pool(ckv, kpe);
    let (hs, rk, sc) = (HEADS as u32, RANK as u32, scale as f32);
    both(
        &mut gpu,
        "mla decode",
        &|g| vec![(g.zeros(n), n, false)],
        std::slice::from_ref(&causal),
        &|g, o| {
            g.run(|c| {
                mla::attention_decode(c, q_t, qpe_t, &cache, pos_t, own_t, hs, rk, sc, o_t(o[0].0))
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::attention_decode(
                    c,
                    st(q_t),
                    st(qpe_t),
                    &sp(&cache),
                    st(pos_t),
                    st(own_t),
                    hs,
                    rk,
                    sc,
                    st(o_t(o[0].0)),
                )
            })
        },
    );
    both(
        &mut gpu,
        "mla prefill",
        &|g| vec![(g.zeros(n), n, false)],
        &[causal],
        &|g, o| {
            g.run(|c| {
                mla::attention_prefill(
                    c,
                    ragged,
                    qpe_t,
                    &cache,
                    pos_t,
                    own_t,
                    hs,
                    rk,
                    sc,
                    o_t(o[0].0),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::attention_prefill(
                    c,
                    sragged,
                    st(qpe_t),
                    &sp(&cache),
                    st(pos_t),
                    st(own_t),
                    hs,
                    rk,
                    sc,
                    st(o_t(o[0].0)),
                )
            })
        },
    );
    both(
        &mut gpu,
        "mla decode selected",
        &|g| vec![(g.zeros(n), n, false)],
        std::slice::from_ref(&selected),
        &|g, o| {
            g.run(|c| {
                mla::attention_decode_selected(
                    c,
                    q_t,
                    qpe_t,
                    sel_t,
                    &cache,
                    pos_t,
                    own_t,
                    hs,
                    rk,
                    sc,
                    o_t(o[0].0),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::attention_decode_selected(
                    c,
                    st(q_t),
                    st(qpe_t),
                    st(sel_t),
                    &sp(&cache),
                    st(pos_t),
                    st(own_t),
                    hs,
                    rk,
                    sc,
                    st(o_t(o[0].0)),
                )
            })
        },
    );
    both(
        &mut gpu,
        "mla prefill selected",
        &|g| vec![(g.zeros(n), n, false)],
        &[selected],
        &|g, o| {
            g.run(|c| {
                mla::attention_prefill_selected(
                    c,
                    ragged,
                    qpe_t,
                    sel_t,
                    &cache,
                    pos_t,
                    own_t,
                    hs,
                    rk,
                    sc,
                    o_t(o[0].0),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::attention_prefill_selected(
                    c,
                    sragged,
                    st(qpe_t),
                    st(sel_t),
                    &sp(&cache),
                    st(pos_t),
                    st(own_t),
                    hs,
                    rk,
                    sc,
                    st(o_t(o[0].0)),
                )
            })
        },
    );

    // latents with rope: the tail turned by its position after the split.
    let tpos = gpu.up_i32(&[0, 3, 11]);
    let theta = 10_000.0f32;
    let rope_want: Vec<f64> = (0..rows * ROPE)
        .map(|x| {
            let (r, d) = (x / ROPE, x % ROPE);
            let half = ROPE / 2;
            let k = d % half;
            let t = f64::from([0, 3, 11][r]) * f64::from(theta).powf(-(k as f64) / half as f64);
            let (a, c) = (ar[r * width + RANK + k], ar[r * width + RANK + k + half]);
            if d < half {
                a * t.cos() - c * t.sin()
            } else {
                a * t.sin() + c * t.cos()
            }
        })
        .collect();
    let mut latent_want = Vec::new();
    for r in 0..rows {
        let lat = &ar[r * width..][..RANK];
        let inv =
            1.0 / (lat.iter().map(|v| v * v).sum::<f64>() / RANK as f64 + f64::from(eps)).sqrt();
        latent_want.extend(lat.iter().zip(&wr).map(|(v, w)| v * inv * w));
    }
    both(
        &mut gpu,
        "mla latents rope",
        &|g| {
            vec![
                (g.zeros(rows * RANK), rows * RANK, false),
                (g.zeros(rows * ROPE), rows * ROPE, false),
            ]
        },
        &[latent_want, rope_want],
        &|g, o| {
            g.run(|c| {
                mla::latents_rope(
                    c,
                    b(kv_a, rows, width),
                    i32t(tpos, rows, 1),
                    b(wn, 1, RANK),
                    eps,
                    RANK as u32,
                    ROPE as u32,
                    theta,
                    b(o[0].0, rows, RANK),
                    b(o[1].0, rows, ROPE),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::mla::latents_rope(
                    c,
                    st(b(kv_a, rows, width)),
                    st(i32t(tpos, rows, 1)),
                    st(b(wn, 1, RANK)),
                    eps,
                    RANK as u32,
                    ROPE as u32,
                    theta,
                    st(b(o[0].0, rows, RANK)),
                    st(b(o[1].0, rows, ROPE)),
                )
            })
        },
    );
}
