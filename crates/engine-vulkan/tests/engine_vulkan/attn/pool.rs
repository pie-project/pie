//! The compressed KV pool on the first Vulkan device, against an f64 host
//! reference: block boundaries, the compressor's state write, the gated
//! pooling of each closed block (with and without a learned position bias),
//! the block mean, the entry store, and the attention over pooled entries
//! with its LSE, causal and selected.

use kernels_slang::{KvPool, RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

const HD: usize = 64;
const RATIO: usize = 4;
const PAGE: usize = 8;

#[test]
fn the_compressed_pool_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let ints = |g: &Gpu, b: u32, n: usize| -> Vec<i32> {
        g.down_f32(b, n)
            .iter()
            .map(|v| v.to_bits() as i32)
            .collect()
    };

    // Request 0 owns pages [1, 0] (positions 0..16), request 1 page [2].
    let page_indices = [1u32, 0, 2];
    let indptr = [0u32, 2, 3];
    let (pi, pp) = (gpu.up_u32(&page_indices), gpu.up_u32(&indptr));
    let slot = |req: usize, pos: usize| {
        page_indices[indptr[req] as usize + pos / PAGE] as usize * PAGE + pos % PAGE
    };
    let slots = 3 * PAGE;
    let pool_of = |keys: u32, width: usize| KvPool {
        keys: t(keys, slots, width, Dtype::Bf16),
        values: t(keys, slots, width, Dtype::Bf16),
        page_indices: t(pi, 3, 1, Dtype::U32),
        page_indptr: t(pp, 3, 1, Dtype::U32),
        page_size: PAGE as i32,
        seq_stride: width as u64,
        head_stride: width as u64,
    };

    // Boundaries: rows at positions 3, 5, 7, 11 of requests 0, 0, 1, 0; the
    // third is not a valid row.
    let positions = [3i32, 5, 7, 11];
    let owner = [0i32, 0, 1, 0];
    let valid = u32::from_le_bytes([1, 1, 0, 1]);
    let (pos, own, vb) = (
        gpu.up_i32(&positions),
        gpu.up_i32(&owner),
        gpu.up_u32(&[valid]),
    );
    let n = positions.len();
    let i32t = |b: u32, r: usize| t(b, r, 1, Dtype::I32);
    let want_pos = [3, -1, -1, 11];
    let want_rope = [0, 0, 0, 8];
    let qp = gpu.up_i32(&[0, 4]);
    for prefill in [false, true] {
        let (bp, br, bro) = (
            gpu.up_i32(&[9; 4]),
            gpu.up_i32(&[9; 4]),
            gpu.up_i32(&[9; 4]),
        );
        let rag = RaggedTensor {
            data: i32t(pos, n),
            indptr: i32t(qp, 2),
        };
        let vt = t(vb, n, 1, Dtype::U8);
        if prefill {
            gpu.run(|c| {
                kernels_slang::attn::pool::boundary_prefill(
                    c,
                    rag,
                    i32t(own, n),
                    vt,
                    RATIO as u32,
                    i32t(bp, n),
                    i32t(br, n),
                    i32t(bro, n),
                )
            });
        } else {
            gpu.run(|c| {
                kernels_slang::attn::pool::boundary_decode(
                    c,
                    i32t(pos, n),
                    i32t(own, n),
                    vt,
                    RATIO as u32,
                    i32t(bp, n),
                    i32t(br, n),
                    i32t(bro, n),
                )
            });
        }
        let who = format!(
            "pool boundary {}",
            if prefill { "prefill" } else { "decode" }
        );
        assert_eq!(ints(&gpu, bp, n), want_pos, "{who}: positions");
        assert_eq!(ints(&gpu, br, n), owner, "{who}: requests");
        assert_eq!(ints(&gpu, bro, n), want_rope, "{who}: rope positions");
        eprintln!("{who}: exact");
    }

    // The state write: coff 2, so rows of 2 x HD into a pitch of 2 x HD.
    let width = 2 * HD;
    let (kv, kvr) = gpu.up(&noise(1, 3 * width));
    let (sc, scr) = gpu.up(&noise(2, 3 * width));
    let (kvr, scr) = (f64s(&kvr), f64s(&scr));
    let (wpg, wof) = ([1u32, 0, 2], [5u32, 7, 0]);
    let (wp, wo) = (gpu.up_u32(&wpg), gpu.up_u32(&wof));
    let mut want_kv = vec![0.0; slots * width];
    let mut want_sc = vec![0.0; slots * width];
    for r in 0..3 {
        let s = wpg[r] as usize * PAGE + wof[r] as usize;
        want_kv[s * width..][..width].copy_from_slice(&kvr[r * width..][..width]);
        want_sc[s * width..][..width].copy_from_slice(&scr[r * width..][..width]);
    }
    let pool_pages = pool_of(pi, HD);
    let u = |b: u32| t(b, 3, 1, Dtype::U32);
    let bt = |b: u32, r: usize, w: usize| t(b, r, w, Dtype::Bf16);
    both(
        &mut gpu,
        "pool state write",
        &|g| {
            vec![
                (g.zeros(slots * width), slots * width, false),
                (g.zeros(slots * width), slots * width, false),
            ]
        },
        &[want_kv, want_sc],
        &|g, o| {
            g.run(|c| {
                kernels_slang::attn::pool::state_write(
                    c,
                    bt(kv, 3, width),
                    bt(sc, 3, width),
                    &pool_pages,
                    u(wp),
                    u(wo),
                    HD as u32,
                    RATIO as u32,
                    bt(o[0].0, slots, width),
                    bt(o[1].0, slots, width),
                )
            })
        },
    );

    // The gather: closed blocks at positions 7 and 15 of request 0, and an
    // open row (-1).
    let bpos = [7i32, 15, -1];
    let breq = [0i32, 0, 1];
    let (bpb, brb) = (gpu.up_i32(&bpos), gpu.up_i32(&breq));
    let (skv, skvr) = gpu.up(&noise(3, slots * width));
    let (ssc, sscr) = gpu.up(&noise(4, slots * width)
        .iter()
        .map(|v| v * 2.0)
        .collect::<Vec<_>>());
    let (skvr, sscr) = (f64s(&skvr), f64s(&sscr));
    for (coff, with_ape) in [(2usize, false), (1, true)] {
        let ape_w = coff * HD;
        let ape_v: Vec<f32> = noise(5, RATIO * ape_w).iter().map(|v| v * 0.5).collect();
        let ape = gpu.up_f32(&ape_v);
        let window = coff * RATIO;
        let want: Vec<f64> = (0..3 * HD)
            .map(|x| {
                let (c, d) = (x / HD, x % HD);
                if bpos[c] < 0 {
                    return 0.0;
                }
                let terms: Vec<(f64, f64)> = (0..window)
                    .filter_map(|i| {
                        let p = bpos[c] + i as i32 - (window as i32 - 1);
                        (p >= 0).then(|| {
                            let p = p as usize;
                            let col = if i >= RATIO { HD } else { 0 } + d;
                            let e = slot(breq[c] as usize, p) * width + col;
                            let bias = if with_ape {
                                f64::from(ape_v[(p % RATIO) * ape_w + col])
                            } else {
                                0.0
                            };
                            (sscr[e] + bias, skvr[e])
                        })
                    })
                    .collect();
                let top = terms.iter().map(|t| t.0).fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = terms.iter().map(|t| (t.0 - top).exp()).sum();
                terms.iter().map(|t| (t.0 - top).exp() * t.1).sum::<f64>() / z
            })
            .collect();
        let apet = with_ape.then(|| t(ape, RATIO, ape_w, Dtype::F32));
        both(
            &mut gpu,
            &format!("pool gather coff {coff} ape {with_ape}"),
            &|g| vec![(g.zeros(3 * HD), 3 * HD, false)],
            &[want],
            &|g, o| {
                g.run(|c| {
                    kernels_slang::attn::pool::gather(
                        c,
                        i32t(bpb, 3),
                        i32t(brb, 3),
                        &pool_pages,
                        HD as u32,
                        RATIO as u32,
                        bt(skv, slots, width),
                        bt(ssc, slots, width),
                        apet,
                        bt(o[0].0, 3, HD),
                    )
                })
            },
        );
    }

    // The block mean of index keys, and the entry store.
    let (ik, ikr) = gpu.up(&noise(6, slots * HD));
    let ikr = f64s(&ikr);
    let key_pool = pool_of(ik, HD);
    let want: Vec<f64> = (0..3 * HD)
        .map(|x| {
            let (c, d) = (x / HD, x % HD);
            if bpos[c] < 0 {
                return 0.0;
            }
            (0..RATIO)
                .filter_map(|i| {
                    let p = bpos[c] + i as i32 - (RATIO as i32 - 1);
                    (p >= 0).then(|| ikr[slot(breq[c] as usize, p as usize) * HD + d])
                })
                .sum::<f64>()
                / RATIO as f64
        })
        .collect();
    both(
        &mut gpu,
        "pool block mean",
        &|g| vec![(g.zeros(3 * HD), 3 * HD, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::attn::index::block_mean(
                    c,
                    i32t(bpb, 3),
                    i32t(brb, 3),
                    &key_pool,
                    HD as u32,
                    RATIO as u32,
                    bt(o[0].0, 3, HD),
                )
            })
        },
    );
    let (ent, entr) = gpu.up(&noise(7, 3 * HD));
    let entr = f64s(&entr);
    let mut want = vec![0.0; slots * HD];
    for c in 0..3 {
        if bpos[c] >= 0 {
            let s = slot(breq[c] as usize, bpos[c] as usize);
            want[s * HD..][..HD].copy_from_slice(&entr[c * HD..][..HD]);
        }
    }
    both(
        &mut gpu,
        "pool store",
        &|g| vec![(g.zeros(slots * HD), slots * HD, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::attn::pool::kv_append(
                    c,
                    bt(ent, 3, HD),
                    i32t(bpb, 3),
                    i32t(brb, 3),
                    &pool_of(o[0].0, HD),
                    u(wp),
                    u(wo),
                )
            })
        },
    );

    // Attention over pooled entries: rows at positions 15, 2 (no closed
    // block), 7 of requests 0, 0, 1; 3 heads.
    let heads = 3usize;
    let apos = [15i32, 2, 7];
    let aown = [0i32, 0, 1];
    let (ap, ao) = (gpu.up_i32(&apos), gpu.up_i32(&aown));
    let (cmp, cmpr) = gpu.up(&noise(8, slots * HD));
    let cmpr = f64s(&cmpr);
    let entries = pool_of(cmp, HD);
    let (q, qr) = gpu.up(&noise(9, 3 * heads * HD));
    let qr = f64s(&qr);
    let scale = 0.125f64;
    let read = |keys: &dyn Fn(usize) -> Vec<usize>| -> (Vec<f64>, Vec<f64>) {
        let mut o = vec![0.0; 3 * heads * HD];
        let mut l = vec![0.0; 3 * heads];
        for r in 0..3 {
            let ks = keys(r);
            for h in 0..heads {
                let qv = &qr[(r * heads + h) * HD..][..HD];
                let rows: Vec<usize> = ks
                    .iter()
                    .map(|&c| slot(aown[r] as usize, (c + 1) * RATIO - 1))
                    .collect();
                let s: Vec<f64> = rows
                    .iter()
                    .map(|&sl| {
                        qv.iter()
                            .zip(&cmpr[sl * HD..][..HD])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                if s.is_empty() {
                    l[r * heads + h] = f64::NEG_INFINITY;
                    continue;
                }
                let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = s.iter().map(|v| (v - top).exp()).sum();
                l[r * heads + h] = (z.ln() + top) * std::f64::consts::LOG2_E;
                for d in 0..HD {
                    o[(r * heads + h) * HD + d] = rows
                        .iter()
                        .zip(&s)
                        .map(|(&sl, v)| (v - top).exp() * cmpr[sl * HD + d])
                        .sum::<f64>()
                        / z;
                }
            }
        }
        (o, l)
    };
    let (causal_o, causal_l) = read(&|r| (0..(apos[r] as usize + 1) / RATIO).collect());
    let sel_v = [3i32, -1, 0, 9, 0, 1, 2, 3, 1, 1, 0, -1];
    let sel = gpu.up_i32(&sel_v);
    let (sel_o, sel_l) = read(&|r| {
        let visible = (apos[r] + 1) / RATIO as i32;
        sel_v[r * 4..][..4]
            .iter()
            .filter(|&&c| c >= 0 && c < visible)
            .map(|&c| c as usize)
            .collect()
    });
    let qt = bt(q, 3, heads * HD);
    let (ot, lt) = (
        |b: u32| bt(b, 3, heads * HD),
        |b: u32| t(b, 3, heads, Dtype::F32),
    );
    let outs = |g: &mut Gpu| {
        vec![
            (g.zeros(3 * heads * HD), 3 * heads * HD, false),
            (g.zeros(2 * 3 * heads), 3 * heads, true),
        ]
    };
    let (hs, hd, rt) = (heads as u32, HD as u32, RATIO as u32);
    both(
        &mut gpu,
        "pool attention lse",
        &outs,
        &[causal_o, causal_l],
        &|g, o| {
            g.run(|c| {
                kernels_slang::attn::pool::attention_lse(
                    c,
                    qt,
                    i32t(ap, 3),
                    i32t(ao, 3),
                    &entries,
                    rt,
                    hs,
                    hd,
                    scale as f32,
                    ot(o[0].0),
                    lt(o[1].0),
                )
            })
        },
    );
    let selt = t(sel, 3, 4, Dtype::I32);
    both(
        &mut gpu,
        "pool attention lse selected",
        &outs,
        &[sel_o, sel_l],
        &|g, o| {
            g.run(|c| {
                kernels_slang::attn::pool::attention_lse_selected(
                    c,
                    qt,
                    i32t(ap, 3),
                    i32t(ao, 3),
                    selt,
                    &entries,
                    rt,
                    4,
                    hs,
                    hd,
                    scale as f32,
                    ot(o[0].0),
                    lt(o[1].0),
                )
            })
        },
    );
}
