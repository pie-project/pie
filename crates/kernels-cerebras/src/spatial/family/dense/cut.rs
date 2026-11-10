//! The unit snake's cut: every unit's weight tiles (`w<u>`: QKV / GDN in, out, gate / up, down a
//! layer, each weight PE's slice), the norm gammas (`gam`), GDN's conv, gate params and norm
//! (`cw`, `gpar`, `gnw`), the q / k norms with q's scale (`qkw`), the RoPE tables (`rot`), the
//! head's vocabulary tiles and final norm (`hw`, `hg`), the embedding (`embed`) and the head's
//! embedding tiles (`hemb`).
//!
//! Column mapping (a weight column c of PX): GDN's k head c / (PX / nk), its DK = hk / (PX / nk)
//! dims, v head c / (PX / nv), DV dims; attention's KV head c / (PX / nkv), Dc = hd / (PX / nkv)
//! dims: the first rd / Dc columns hold Dc / 2 rotated pairs (i, i + rd / 2) side by side, the
//! others Dc unrotated dims; Qh = nq / nkv q heads (with their gates) a column. Where PY does not
//! divide the hidden size, Hc = ceil(H / PY) and the padded dims carry zero weights, gammas and
//! embedding.

use crate::spatial::plan::{LayerKind, Plan};
use crate::spatial::tiles::spec::{
    Elem, File, Fill, Gather, Grid, Head, Map, Part, Rope, Spec, padded, range,
};

type R<T> = Result<T, String>;

pub fn tiles(plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
    let has = |n: &str| shapes(n).is_some();
    let (s, p) = (&plan.shape, &plan.partition);
    let (px, py) = (p.cols as usize, p.rows as usize);
    let k = p.layers_per_unit as usize;
    let nu = p.units as usize;
    let (hr, vr) = (p.head.rows as usize, p.head.vocab_rows as usize);
    let (rlv, rbits) = (
        plan.contract.rope_levels as usize,
        plan.contract.rope_bits as usize,
    );
    let h = s.hidden as usize;
    let v = s.vocab as usize;
    let nl = s.layers.len();
    let gate = s.attn.output_gate;
    let one = if s.norm.unit_offset {
        Map::PlusOne
    } else {
        Map::Id
    };
    let (nq, nkv) = (s.attn.q_heads as usize, s.attn.kv_heads as usize);
    let hd = s.attn.head_dim as usize;
    let (nk, nv, hk, hv) = match &s.gdn {
        Some(g) => (
            g.key_heads as usize,
            g.value_heads as usize,
            g.key_dim as usize,
            g.value_dim as usize,
        ),
        None => (nkv, nkv, hd, nq / nkv * hd),
    };
    let rd = s.rope.rotary_dims as usize;
    let kinds: Vec<bool> = s
        .layers
        .iter()
        .map(|l| matches!(l, LayerKind::Gdn))
        .collect();
    if nl != nu * k {
        return Err(format!("dense: {nl} layers in {nu} units of {k}"));
    }
    // a layer's param by its role: the first of the names the traces give it
    let name = |l: usize, names: &[&str]| -> Option<String> {
        names
            .iter()
            .map(|n| format!("layers.{l}.{n}"))
            .find(|n| has(n))
    };
    let need = |l: usize, names: &[&str]| -> R<String> {
        name(l, names).ok_or_else(|| format!("dense: no param layers.{l}.{}", names.join(" / ")))
    };
    let pat = &kinds[..k];
    let nlg = pat.iter().filter(|g| **g).count();
    // GDN's conv taps: its param's width
    let conv = match kinds.iter().position(|g| *g) {
        Some(l) => shapes(&need(l, &["gdn.conv"])?).map_or(4, |s| s[1] as usize),
        None => 4,
    };
    let inter = s.mlp.intermediate as usize;
    if !inter.is_multiple_of(px) {
        return Err(format!("dense: the MLP's {inter} over {px} columns"));
    }
    let hc = h.div_ceil(py);
    let ic = inter / px;
    let (cpk, cpv, cpa) = (px / nk, px / nv, px / nkv);
    let (dk, dv, dc, qh) = (hk / cpk, hv / cpv, hd / cpa, nq / nkv);
    let nrot = dc / 2;
    let nrc = rd / dc;
    let (hcols, hdc) = (py / 2, 2 * hc);
    let (kd, vd) = (nk * hk, nv * hv);

    let cols = |f: &dyn Fn(usize) -> Vec<i64>| -> Vec<Vec<i64>> { (0..px).map(f).collect() };
    let qidx = |c: usize| -> Vec<i64> {
        (0..dk)
            .map(|j| ((c / cpk) * hk + (c % cpk) * dk + j) as i64)
            .collect()
    };
    let vidx = |c: usize| -> Vec<i64> {
        (0..dv)
            .map(|j| ((c / cpv) * hv + (c % cpv) * dv + j) as i64)
            .collect()
    };
    // a column's head dims: rotated pairs side by side, then plain dims
    let perm = |pa: usize| -> Vec<i64> {
        if pa < nrc {
            let h2 = rd / 2;
            (0..nrot)
                .flat_map(|j| [(pa * nrot + j) as i64, (pa * nrot + j + h2) as i64])
                .collect()
        } else {
            (0..dc).map(|j| (rd + (pa - nrc) * dc + j) as i64).collect()
        }
    };
    let pd: Vec<Vec<i64>> = (0..px).map(|c| perm(c % cpa)).collect();
    let nch = 2 * dk + dv;
    let (gw, cww, gpw, gnww, qkww) = (
        2 * k * hc,
        nlg.max(1) * conv * nch,
        nlg.max(1) * 2,
        nlg.max(1) * dv,
        (k - nlg).max(1) * (qh * dc + dc),
    );
    let isq = (hd as f64).sqrt() as f32;
    let (qmap, kmap) = match one {
        Map::PlusOne => (Map::PlusOneDiv(isq), Map::PlusOne),
        _ => (Map::Div(isq), Map::Id),
    };
    let rows: Vec<(usize, usize)> = (0..py).map(|r| (r * hc, hc)).collect();
    let mut gam = File::new("gam", vec![nu, py, gw], Elem::F16);
    let mut cw = File::new("cw", vec![nu, px, cww], Elem::F32);
    let mut gpar = File::new("gpar", vec![nu, px, gpw], Elem::F32);
    let mut gnw = File::new("gnw", vec![nu, px, gnww], Elem::F32);
    let mut qkw = File::new("qkw", vec![nu, px, qkww], Elem::F32);
    let mut spec = Spec::default();
    for u in 0..nu {
        let mut parts = Vec::new();
        let (mut g, mut a) = (0usize, 0usize);
        for (l, &gdn) in kinds.iter().enumerate().skip(k * u).take(k) {
            let lk = l % k;
            let ln1 = need(l, &["mixer_norm", "attn_norm"])?;
            let ln2 = need(l, &["mlp_norm"])?;
            for r in 0..py {
                let idx = || {
                    padded(
                        (0..hc)
                            .map(|i| r * hc + i)
                            .filter(|d| *d < h)
                            .map(|d| d as i64),
                        hc,
                    )
                };
                let at = (u * py + r) * gw + lk * 2 * hc;
                gam.fills.push(Fill::Gather(Gather {
                    at,
                    src: ln1.clone(),
                    idx: idx(),
                    map: one,
                }));
                gam.fills.push(Fill::Gather(Gather {
                    at: at + hc,
                    src: ln2.clone(),
                    idx: idx(),
                    map: one,
                }));
            }
            if gdn {
                // GDN: [qkv | z | b | a] in, out
                let idx = cols(&|c| {
                    let (q, vv, vh) = (qidx(c), vidx(c), (c / cpv) as i64);
                    let mut o = q.clone();
                    o.extend(q.iter().map(|x| kd as i64 + x));
                    o.extend(vv.iter().map(|x| (2 * kd) as i64 + x));
                    o.extend(vv.iter().map(|x| (2 * kd + vd) as i64 + x));
                    o.push((2 * kd + 2 * vd) as i64 + vh);
                    o.push((2 * kd + 2 * vd + nv) as i64 + vh);
                    o
                });
                parts.push(Part::In {
                    src: vec![need(l, &["gdn.in_qkvz"])?, need(l, &["gdn.in_ba"])?],
                    outs: idx,
                });
                parts.push(Part::Out {
                    src: vec![need(l, &["gdn.out_proj"])?],
                    ins: cols(&vidx),
                });
                let (cv, al, dt, gn) = (
                    need(l, &["gdn.conv"])?,
                    need(l, &["gdn.a_log"])?,
                    need(l, &["gdn.dt_bias"])?,
                    need(l, &["gdn.norm"])?,
                );
                for c in 0..px {
                    let (q, vv) = (qidx(c), vidx(c));
                    let ch: Vec<i64> = q
                        .iter()
                        .copied()
                        .chain(q.iter().map(|x| kd as i64 + x))
                        .chain(vv.iter().map(|x| (2 * kd) as i64 + x))
                        .collect();
                    let base = (u * px + c) * cww + g * conv * nch;
                    let idx = (0..conv)
                        .flat_map(|t| ch.iter().map(move |chn| chn * conv as i64 + t as i64))
                        .collect();
                    cw.fills.push(Fill::Gather(Gather {
                        at: base,
                        src: cv.clone(),
                        idx,
                        map: Map::Id,
                    }));
                    let vh = (c / cpv) as i64;
                    let gp = (u * px + c) * gpw + 2 * g;
                    gpar.fills.push(Fill::Gather(Gather {
                        at: gp,
                        src: al.clone(),
                        idx: vec![vh],
                        map: Map::Id,
                    }));
                    gpar.fills.push(Fill::Gather(Gather {
                        at: gp + 1,
                        src: dt.clone(),
                        idx: vec![vh],
                        map: Map::Id,
                    }));
                    gnw.fills.push(Fill::Gather(Gather {
                        at: (u * px + c) * gnww + g * dv,
                        src: gn.clone(),
                        idx: range((c % cpv) * dv, dv),
                        map: Map::Id,
                    }));
                }
                g += 1;
            } else {
                let qrow = ((1 + usize::from(gate)) * hd) as i64;
                let qr = nq as i64 * qrow;
                let (hd_, nkv_) = (hd as i64, nkv as i64);
                let idx = cols(&|c| {
                    let kvh = (c / cpa) as i64;
                    let mut o = Vec::new();
                    for x in 0..qh as i64 {
                        o.extend(pd[c].iter().map(|d| (qh as i64 * kvh + x) * qrow + d));
                    }
                    if gate {
                        for x in 0..qh as i64 {
                            o.extend(
                                pd[c]
                                    .iter()
                                    .map(|d| (qh as i64 * kvh + x) * 2 * hd_ + hd_ + d),
                            );
                        }
                    }
                    o.extend(pd[c].iter().map(|d| qr + kvh * hd_ + d));
                    o.extend(pd[c].iter().map(|d| qr + nkv_ * hd_ + kvh * hd_ + d));
                    o
                });
                parts.push(Part::In {
                    src: vec![need(l, &["attn.qg_proj_k_proj_v_proj", "qkv"])?],
                    outs: idx,
                });
                let oidx = cols(&|c| {
                    let kvh = (c / cpa) as i64;
                    (0..qh as i64)
                        .flat_map(|x| pd[c].iter().map(move |d| (qh as i64 * kvh + x) * hd_ + d))
                        .collect()
                });
                parts.push(Part::Out {
                    src: vec![need(l, &["attn.o_proj", "o"])?],
                    ins: oidx,
                });
                let qn = name(l, &["attn.q_norm", "q_norm"]);
                let kn = name(l, &["attn.k_norm", "k_norm"]);
                for (c, dims) in pd.iter().enumerate() {
                    let base = (u * px + c) * qkww + a * (qh * dc + dc);
                    for x in 0..qh {
                        let at = base + x * dc;
                        qkw.fills.push(match &qn {
                            Some(q) => Fill::Gather(Gather {
                                at,
                                src: q.clone(),
                                idx: dims.clone(),
                                map: qmap,
                            }),
                            None => Fill::Const {
                                at: (at..at + dc).collect(),
                                value: 1.0 / isq,
                            },
                        });
                    }
                    let at = base + qh * dc;
                    qkw.fills.push(match &kn {
                        Some(k) => Fill::Gather(Gather {
                            at,
                            src: k.clone(),
                            idx: dims.clone(),
                            map: kmap,
                        }),
                        None => Fill::Const {
                            at: (at..at + dc).collect(),
                            value: 1.0,
                        },
                    });
                }
                a += 1;
            }
            let gi = |c: usize| -> Vec<i64> { range(c * ic, ic) };
            let idx = cols(&|c| {
                let mut o = gi(c);
                o.extend(gi(c).iter().map(|x| inter as i64 + x));
                o
            });
            parts.push(Part::In {
                src: vec![need(l, &["mlp.gate_up", "gate_up"])?],
                outs: idx,
            });
            parts.push(Part::Out {
                src: vec![need(l, &["mlp.down", "down"])?],
                ins: cols(&gi),
            });
        }
        let w = parts.iter().map(|q| q.width(hc)).sum();
        spec.push(
            File::new(format!("w{u}"), vec![py, px, w], Elem::F16).with(Fill::Grid(Grid {
                hidden: h,
                rows: rows.clone(),
                parts,
            })),
        );
    }
    for f in [gam, cw, gpar, gnw, qkw] {
        spec.push(f);
    }

    // RoPE tables: a position is RLV digits of RBITS bits
    let inv: Vec<f64> = match &s.rope.freqs {
        // the plan's frequencies as its JSON states them
        Some(f) => f
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap_or(f64::from(*x)))
            .collect(),
        None => s.rope.inv_freq(s.attn.head_dim),
    };
    let freqs = (0..px)
        .map(|pc| {
            vec![
                (0..nrot)
                    .map(|j| {
                        if pc % cpa < nrc {
                            inv[(pc % cpa) * nrot + j]
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ]
        })
        .collect();
    let rb = 1usize << rbits;
    spec.push(
        File::new("rot", vec![px, rlv * rb * 2 * nrot], Elem::F32).with(Fill::Rope(Rope {
            freqs,
            levels: rlv,
            radix: rb,
            scale0: 1.0,
        })),
    );

    // the head: rows past the vocabulary are copies of its last row (their logits tie with it
    // and the argmax keeps the real, lower id)
    let head = if !s.head.tied && has("head") {
        "head"
    } else {
        "embed"
    };
    let dims: Vec<Vec<i64>> = (0..hcols)
        .map(|hcol| {
            padded(
                (0..hdc)
                    .map(|kk| hcol * hdc + kk)
                    .filter(|d| *d < h)
                    .map(|d| d as i64),
                hdc,
            )
        })
        .collect();
    let tiles = |src: &str, row_major: bool, copy_last: bool| Head {
        src: src.into(),
        vocab: v,
        groups: hr,
        per: vr,
        dims: dims.clone(),
        dims_major: false,
        row_major,
        copy_last,
        scale: 1.0,
    };
    spec.push(
        File::new("hw", vec![hr, hcols, hdc * vr], Elem::F16)
            .with(Fill::Head(tiles(head, false, true))),
    );
    spec.push(
        File::new("hg", vec![hcols, hdc], Elem::F32).with(Fill::Copy {
            at: 0,
            src: "final_norm".into(),
            map: one,
        }),
    );
    spec.push(File::new("embed", vec![v, h], Elem::Bf16).with(Fill::Copy {
        at: 0,
        src: "embed".into(),
        map: Map::Id,
    }));
    spec.push(
        File::new("hemb", vec![hr, hcols, vr * hdc], Elem::Bf16)
            .with(Fill::Head(tiles("embed", true, false))),
    );
    Ok(spec)
}
