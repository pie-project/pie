//! Gemma 4 12B's cut: units of K layers (S .. S G), unit u = layers K u .. K u + K - 1.
//! - Rows: attention rows r < RA hold Hc A hidden dims, plain rows Hc W; row r's dims are
//!   [off_r, off_r + Hc_r) of the padded hidden (dims >= H: zero weights, zero gammas).
//! - Columns: segment s = c / SWA, p = c % SWA. Sliding: kv head s, q heads Qh s ..; global: the
//!   kv head replicated (every segment a copy), q heads Qh s ... Within the segment, Dc / 2
//!   rotated pairs (j, j + hd / 2), j = p Dc / 2 + t, side by side; j >= hd / 2: padding (zero
//!   weights, identity RoPE, zero norm weights).
//! - MLP: column c holds intermediate dims c Ic .. (>= I: padding).
//! - Head: column c = the dims of its row group, HDc slots (padding past them); tied, f16.
//!
//! `w<u>a` / `w<u>w` (f16, the A / W rows), `gama` / `gamw` (f32: a layer's input /
//! post-attention / pre-FFN / post-FFN gammas, the layer scalar; + the embedding scale), `qkw`,
//! `rot`, `hw0` / `hw1` (the head's halves: one file is over the 2 GiB a stage takes), `hg`,
//! `embed`.

use crate::plan::{LayerKind, Plan};
use crate::tiles::spec::{Elem, File, Fill, Gather, Grid, Head, Map, Part, Rope, Spec, padded};

type R<T> = Result<T, String>;

pub fn tiles(plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
    let (s, p) = (&plan.shape, &plan.partition);
    let (px, py) = (p.cols as usize, p.rows as usize);
    let (ra, swa) = (p.kv_rows as usize, p.segments.scores as usize);
    let (hca, hcw) = (p.row_dims[0].dims as usize, p.row_dims[1].dims as usize);
    let (h, v, nl) = (s.hidden as usize, s.vocab as usize, s.layers.len());
    let (nq, nkv) = (s.attn.q_heads as usize, s.attn.kv_heads as usize);
    let hd_s = s.attn.head_dim as usize;
    let hd_g = s.attn.global_head_dim.map_or(hd_s, |d| d as usize);
    let (nu, k) = (p.units as usize, p.layers_per_unit as usize);
    let nseg = px / swa;
    if nl != k * nu || nseg * swa != px || nkv != nseg || !nq.is_multiple_of(nseg) {
        return Err(format!(
            "gemma12: {nl} layers in {nu} units of {k}, {nkv} KV heads on {nseg} segments"
        ));
    }
    let qh = nq / nseg;
    let dcf = |hd: usize| 2 * (hd / 2).div_ceil(swa);
    let (dc_s, dc_g) = (dcf(hd_s), dcf(hd_g));
    let inter = s.mlp.intermediate as usize;
    let ic = inter.div_ceil(px);
    let hc: Vec<usize> = (0..py).map(|r| if r < ra { hca } else { hcw }).collect();
    let off: Vec<usize> = std::iter::once(0)
        .chain(hc.iter().scan(0, |a, d| {
            *a += d;
            Some(*a)
        }))
        .collect();
    let hp = off[py];
    if hp < h {
        return Err(format!("gemma12: {hp} hidden slots for {h}"));
    }
    let global = |l: usize| matches!(s.layers[l], LayerKind::Global);
    let pdims = |hd: usize, dc: usize, c: usize| -> Vec<i64> {
        let (nr, pa) = (dc / 2, c % swa);
        (0..nr)
            .flat_map(|t| {
                let j = pa * nr + t;
                if j < hd / 2 {
                    [j as i64, (j + hd / 2) as i64]
                } else {
                    [-1, -1]
                }
            })
            .collect()
    };
    let cols = |f: &dyn Fn(usize) -> Vec<i64>| -> Vec<Vec<i64>> { (0..px).map(f).collect() };
    let lname = |l: usize, n: &str| format!("layers.{l}.{n}");
    let shift = |o: i64, d: &i64| if *d < 0 { -1 } else { o + d };

    let (gla, glw) = (4 * hca + 1, 4 * hcw + 1);
    let qks = qh * dc_g + dc_g;
    let esc = (h as f32).sqrt();
    let mut gama = File::new("gama", vec![nu, ra, k * gla + 1], Elem::F32);
    let mut gamw = File::new("gamw", vec![nu, py - ra, k * glw + 1], Elem::F32);
    let mut qkw = File::new("qkw", vec![nu, px, k * qks], Elem::F32);
    gama.fills.push(Fill::Const {
        at: (0..nu * ra).map(|i| i * (k * gla + 1) + k * gla).collect(),
        value: esc,
    });
    gamw.fills.push(Fill::Const {
        at: (0..nu * (py - ra))
            .map(|i| i * (k * glw + 1) + k * glw)
            .collect(),
        value: esc,
    });
    let dims = |lo: usize, n: usize| padded((lo..lo + n).filter(|d| *d < h).map(|d| d as i64), n);
    let mut spec = Spec::default();
    for u in 0..nu {
        let mut parts = Vec::new();
        for ll in 0..k {
            let l = k * u + ll;
            let (hd, dc) = if global(l) {
                (hd_g, dc_g)
            } else {
                (hd_s, dc_s)
            };
            let norms = [
                "attn_norm",
                "post_attn_norm",
                "pre_ffw_norm",
                "post_ffw_norm",
            ];
            // a band's rows: its gammas (a layer's GL = 4 Hc + 1 a row: the norms, the scalar)
            for (gam, lo, hi, dims_r) in [(&mut gama, 0, ra, hca), (&mut gamw, ra, py, hcw)] {
                let gl = 4 * dims_r + 1;
                for (rr, &o) in off[lo..hi].iter().enumerate() {
                    let at = ((u * (hi - lo) + rr) * (k * gl + 1)) + ll * gl;
                    for (i, n) in norms.iter().enumerate() {
                        gam.fills.push(Fill::Gather(Gather {
                            at: at + i * dims_r,
                            src: lname(l, n),
                            idx: dims(o, dims_r),
                            map: Map::Id,
                        }));
                    }
                    gam.fills.push(Fill::Gather(Gather {
                        at: at + 4 * dims_r,
                        src: lname(l, "scalar"),
                        idx: vec![0],
                        map: Map::Id,
                    }));
                }
            }
            let qrows = |c: usize| -> Vec<i64> {
                let seg = c / swa;
                (0..qh)
                    .flat_map(|x| {
                        pdims(hd, dc, c)
                            .into_iter()
                            .map(move |d| shift(((qh * seg + x) * hd) as i64, &d))
                    })
                    .collect()
            };
            let (nq_hd, nkv_hd) = ((nq * hd) as i64, (nkv * hd) as i64);
            let idx = cols(&|c| {
                let kvo = if global(l) {
                    0
                } else {
                    ((c / swa) * hd) as i64
                };
                let krows: Vec<i64> = pdims(hd, dc, c).iter().map(|d| shift(kvo, d)).collect();
                let mut o = qrows(c);
                o.extend(krows.iter().map(|d| shift(nq_hd, d)));
                if !global(l) {
                    o.extend(krows.iter().map(|d| shift(nq_hd + nkv_hd, d)));
                }
                o
            });
            parts.push(Part::In {
                src: vec![lname(l, "attn.banks.qkv")],
                outs: idx,
            });
            parts.push(Part::Out {
                src: vec![lname(l, "o_proj")],
                ins: cols(&qrows),
            });
            for c in 0..px {
                let base = (u * px + c) * k * qks + ll * qks;
                let pd = pdims(hd, dc, c);
                for x in 0..qh {
                    qkw.fills.push(Fill::Gather(Gather {
                        at: base + x * dc,
                        src: lname(l, "attn.q_norm"),
                        idx: pd.clone(),
                        map: Map::Id,
                    }));
                }
                qkw.fills.push(Fill::Gather(Gather {
                    at: base + qh * dc,
                    src: lname(l, "attn.banks.k_norm"),
                    idx: pd,
                    map: Map::Id,
                }));
            }
            let gi = |c: usize| -> Vec<i64> {
                (0..ic)
                    .map(|j| c * ic + j)
                    .map(|d| if d < inter { d as i64 } else { -1 })
                    .collect()
            };
            parts.push(Part::In {
                src: vec![lname(l, "gate_up")],
                outs: cols(&|c| {
                    let mut o = gi(c);
                    o.extend(gi(c).iter().map(|d| shift(inter as i64, d)));
                    o
                }),
            });
            parts.push(Part::Out {
                src: vec![lname(l, "down")],
                ins: cols(&gi),
            });
        }
        for (tag, lo, hi, dims) in [("a", 0, ra, hca), ("w", ra, py, hcw)] {
            let lw = parts.iter().map(|q| q.width(dims)).sum();
            spec.push(
                File::new(format!("w{u}{tag}"), vec![hi - lo, px, lw], Elem::F16).with(Fill::Grid(
                    Grid {
                        hidden: h,
                        rows: (lo..hi).map(|r| (off[r], hc[r])).collect(),
                        parts: parts.clone(),
                    },
                )),
            );
        }
    }
    spec.push(gama);
    spec.push(gamw);
    spec.push(qkw);

    // RoPE tables a column: [kind (sliding, global)][levels][digits][cos, sin][NPM pairs]
    let npm = dc_g / 2;
    let kinds = [
        (hd_s, dc_s, &s.rope),
        (hd_g, dc_g, s.rope.global.as_deref().unwrap_or(&s.rope)),
    ];
    let freqs = (0..px)
        .map(|c| {
            kinds
                .iter()
                .map(|(hd, dc, rope)| {
                    let inv = rope.inv_freq(*hd as u32);
                    let nr = dc / 2;
                    (0..npm)
                        .map(|t| {
                            let j = (c % swa) * nr + t;
                            if t < nr && j < hd / 2 && j < inv.len() {
                                f64::from(inv[j] as f32)
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let (levels, radix) = (
        plan.contract.rope_levels as usize,
        1usize << plan.contract.rope_bits,
    );
    spec.push(
        File::new("rot", vec![px, 2 * levels * radix * 2 * npm], Elem::F32).with(Fill::Rope(
            Rope {
                freqs,
                levels,
                radix,
                scale0: 1.0,
            },
        )),
    );

    // the head (tied): column c = its group's dims; padded vocabulary rows copy the last real row
    let groups = p.head.groups.clone().unwrap_or_default();
    let (hr, vr) = (p.head.rows as usize, p.head.vocab_rows as usize);
    let hdc = groups
        .iter()
        .map(|g| g.iter().map(|&r| hc[r as usize]).sum::<usize>())
        .max()
        .unwrap_or(0);
    if hr * vr < v {
        return Err(format!("gemma12: a head of {hr} x {vr} rows"));
    }
    let hdims: Vec<Vec<i64>> = groups
        .iter()
        .map(|g| {
            padded(
                g.iter()
                    .flat_map(|&r| off[r as usize]..off[r as usize + 1])
                    .map(|d| if d < h { d as i64 } else { -1 }),
                hdc,
            )
        })
        .collect();
    let hxc = groups.len() / 2;
    for (i, (c0, c1)) in [(0, hxc), (hxc, groups.len())].into_iter().enumerate() {
        spec.push(
            File::new(format!("hw{i}"), vec![hr, c1 - c0, hdc * vr], Elem::F16).with(Fill::Head(
                Head {
                    src: "embed".into(),
                    vocab: v,
                    groups: hr,
                    per: vr,
                    dims: hdims[c0..c1].to_vec(),
                    dims_major: false,
                    row_major: false,
                    copy_last: true,
                    scale: 1.0,
                },
            )),
        );
    }
    spec.push(
        File::new("hg", vec![groups.len(), hdc], Elem::F32).with(Fill::Gather(Gather {
            at: 0,
            src: "final_norm".into(),
            idx: hdims.concat(),
            map: Map::Id,
        })),
    );
    spec.push(File::new("embed", vec![v, h], Elem::Bf16).with(Fill::Copy {
        at: 0,
        src: "embed".into(),
        map: Map::Id,
    }));
    let _ = shapes;
    Ok(spec)
}
