//! Gemma 4 E2B / E4B's cut: units of K layers (S .. S G), unit u = layers K u .. K u + K - 1;
//! units past the KV owners hold shared-KV layers (q only). Weight column c (of PX): kv copy
//! c / (PX / 2) (its Qh = nq / 2 q heads), within the copy p = c % (PX / 2): Dc / 2 rotated pairs
//! (j, j + hd / 2), j = p Dc / 2 + t, side by side; PLE dim c.
//!
//! `w<u>` (f16: a layer's QKV in, o out, gate / up in, down out, PLE gate in, PLE projection
//! out), `gam` (f32: a layer's input / post-attention / pre-FFN / post-FFN / post-PLE gammas, the
//! layer scalar; + the embedding scale), `qkw`, `rot`, `hw`, `hg`, `hemb`, `embed`, and the
//! per-layer inputs of every vocabulary id in each PLE storage column's files
//! (`pt<col>_<part>`).

use crate::plan::{LayerKind, Plan};
use crate::tiles::spec::{Elem, File, Fill, Gather, Grid, Head, Map, Part, Ple, Rope, Spec, range};

use crate::block::ple::FILE_MAX;
use crate::compose::stacked::Arrangement;

type R<T> = Result<T, String>;

pub fn tiles(plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
    let (s, p) = (&plan.shape, &plan.partition);
    let has = |n: &str| shapes(n).is_some();
    let (px, py) = (p.cols as usize, p.rows as usize);
    let (h, v) = (s.hidden as usize, s.vocab as usize);
    let nl = s.layers.len();
    let (nq, nkv) = (s.attn.q_heads as usize, s.attn.kv_heads as usize);
    let hd_s = s.attn.head_dim as usize;
    let hd_g = s.attn.global_head_dim.map_or(hd_s, |d| d as usize);
    let nu = p.units as usize;
    let k = nl / nu;
    let ple_w = s.ple.ok_or("gemma: no per-layer inputs")?.width as usize;
    if nl != k * nu || ple_w != px || !(1..=2).contains(&nkv) {
        return Err(format!(
            "gemma: {nl} layers in {nu} units, PLE {ple_w} on {px} columns, {nkv} KV heads"
        ));
    }
    let hc = h / py;
    let cpa = px / 2;
    let qh = nq / 2;
    let (dc_s, dc_g) = (hd_s / cpa, hd_g / cpa);
    let global = |l: usize| {
        matches!(
            s.layers[l],
            LayerKind::Global | LayerKind::SharedKv { sliding: false, .. }
        )
    };
    let shared = |l: usize| matches!(s.layers[l], LayerKind::SharedKv { .. });
    let pdims = |hd: usize, dc: usize, c: usize| -> Vec<i64> {
        let (nr, pa) = (dc / 2, c % cpa);
        (0..nr)
            .flat_map(|t| [(pa * nr + t) as i64, (pa * nr + t + hd / 2) as i64])
            .collect()
    };
    let cols = |f: &dyn Fn(usize) -> Vec<i64>| -> Vec<Vec<i64>> { (0..px).map(f).collect() };
    let lname = |l: usize, n: &str| format!("layers.{l}.{n}");
    let pname = |l: usize, n: &str| format!("ple.per_layer.{l}.{n}");
    let scalar = |l: usize| -> R<String> {
        [pname(l, "scalar"), lname(l, "scalar")]
            .into_iter()
            .find(|n| has(n))
            .ok_or_else(|| format!("gemma: no layer scalar of layer {l}"))
    };

    let gl = 5 * hc + 1;
    let qks = qh * dc_g + dc_g;
    let rows: Vec<(usize, usize)> = (0..py).map(|r| (r * hc, hc)).collect();
    let mut gam = File::new("gam", vec![nu, py, k * gl + 1], Elem::F32);
    let mut qkw = File::new("qkw", vec![nu, px, k * qks], Elem::F32);
    gam.fills.push(Fill::Const {
        at: (0..nu * py).map(|i| i * (k * gl + 1) + k * gl).collect(),
        value: (h as f32).sqrt(),
    });
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
                lname(l, "attn_norm"),
                lname(l, "post_attn_norm"),
                lname(l, "pre_ffw_norm"),
                lname(l, "post_ffw_norm"),
                pname(l, "norm"),
            ];
            for r in 0..py {
                let at = (u * py + r) * (k * gl + 1) + ll * gl;
                for (i, n) in norms.iter().enumerate() {
                    gam.fills.push(Fill::Gather(Gather {
                        at: at + i * hc,
                        src: n.clone(),
                        idx: range(r * hc, hc),
                        map: Map::Id,
                    }));
                }
                gam.fills.push(Fill::Gather(Gather {
                    at: at + 5 * hc,
                    src: scalar(l)?,
                    idx: vec![0],
                    map: Map::Id,
                }));
            }
            let qrows = |c: usize| -> Vec<i64> {
                let cp = c / cpa;
                (0..qh)
                    .flat_map(|x| {
                        pdims(hd, dc, c)
                            .into_iter()
                            .map(move |d| ((qh * cp + x) * hd) as i64 + d)
                    })
                    .collect()
            };
            let (nq_hd, nkv_hd) = ((nq * hd) as i64, (nkv * hd) as i64);
            if shared(l) {
                parts.push(Part::In {
                    src: vec![lname(l, "attn.banks.q_proj")],
                    outs: cols(&qrows),
                });
            } else {
                let idx = cols(&|c| {
                    // the column's kv head (2 kv heads: its half's)
                    let kvo = if nkv == 2 { ((c / cpa) * hd) as i64 } else { 0 };
                    let pd = pdims(hd, dc, c);
                    let mut o = qrows(c);
                    o.extend(pd.iter().map(|d| nq_hd + kvo + d));
                    o.extend(pd.iter().map(|d| nq_hd + nkv_hd + kvo + d));
                    o
                });
                parts.push(Part::In {
                    src: vec![lname(l, "attn.banks.qkv")],
                    outs: idx,
                });
            }
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
                if !shared(l) {
                    qkw.fills.push(Fill::Gather(Gather {
                        at: base + qh * dc,
                        src: lname(l, "attn.banks.k_norm"),
                        idx: pd,
                        map: Map::Id,
                    }));
                }
            }
            let gu = lname(l, "gate_up");
            let ii = shapes(&gu).ok_or_else(|| format!("gemma: no param {gu}"))?[0] as usize / 2;
            let ic = ii / px;
            if ic * px != ii {
                return Err(format!("gemma: layer {l}'s MLP {ii} over {px} columns"));
            }
            let gi = |c: usize| range(c * ic, ic);
            parts.push(Part::In {
                src: vec![gu],
                outs: cols(&|c| {
                    let mut o = gi(c);
                    o.extend(gi(c).iter().map(|x| ii as i64 + x));
                    o
                }),
            });
            parts.push(Part::Out {
                src: vec![lname(l, "down")],
                ins: cols(&gi),
            });
            parts.push(Part::In {
                src: vec![pname(l, "gate")],
                outs: cols(&|c| vec![c as i64]),
            });
            parts.push(Part::Out {
                src: vec![pname(l, "proj")],
                ins: cols(&|c| vec![c as i64]),
            });
        }
        let lw = parts.iter().map(|q| q.width(hc)).sum();
        spec.push(
            File::new(format!("w{u}"), vec![py, px, lw], Elem::F16).with(Fill::Grid(Grid {
                hidden: h,
                rows: rows.clone(),
                parts,
            })),
        );
    }
    spec.push(gam);
    spec.push(qkw);

    // RoPE tables a column: [kind (sliding, global)][levels][digits][cos, sin][2 pairs]
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
                    (0..2)
                        .map(|t| {
                            let j = (c % cpa) * nr + t;
                            if t < nr && j < inv.len() {
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
        File::new("rot", vec![px, 2 * levels * radix * 2 * 2], Elem::F32).with(Fill::Rope(Rope {
            freqs,
            levels,
            radix,
            scale0: 1.0,
        })),
    );

    // the head (tied): padded vocabulary rows copy the last real row, so they tie it and the
    // lower (real) id wins
    let (hcols, hdc, hr, vr) = (
        p.head.cols as usize,
        2 * hc,
        p.head.rows as usize,
        p.head.vocab_rows as usize,
    );
    if hcols * hdc != h || hr * vr < v {
        return Err(format!(
            "gemma: a head of {hcols} x {hdc} dims, {hr} x {vr} rows"
        ));
    }
    let dims: Vec<Vec<i64>> = (0..hcols).map(|j| range(j * hdc, hdc)).collect();
    let tiles = |row_major: bool, copy_last: bool| Head {
        src: "embed".into(),
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
        File::new("hw", vec![hr, hcols, hdc * vr], Elem::F16).with(Fill::Head(tiles(false, true))),
    );
    spec.push(
        File::new("hemb", vec![hr, hcols, vr * hdc], Elem::Bf16)
            .with(Fill::Head(tiles(true, false))),
    );
    spec.push(
        File::new("hg", vec![hcols, hdc], Elem::F32).with(Fill::Copy {
            at: 0,
            src: "final_norm".into(),
            map: Map::Id,
        }),
    );
    spec.push(File::new("embed", vec![v, h], Elem::Bf16).with(Fill::Copy {
        at: 0,
        src: "embed".into(),
        map: Map::Id,
    }));

    // the per-layer inputs, every vocab id, f16, into each storage column's files
    let arr = Arrangement::of(plan)?;
    let tables: Vec<String> = (0..nl).map(|l| pname(l, "table")).collect();
    for (ci, (_, units, st)) in arr.columns(p.cols + 1).iter().enumerate() {
        let (nr, nt) = (st.rows as usize, st.tokens as usize);
        let layers: Vec<usize> = units
            .iter()
            .flat_map(|&u| (0..k).map(move |ll| k * u as usize + ll))
            .collect();
        let lv = layers.len();
        if nr * nt < v {
            return Err(format!(
                "gemma: PLE column {ci} holds {} of {v} ids",
                nr * nt
            ));
        }
        let per = (FILE_MAX as usize / (px * nt * lv * 2)).max(1);
        for (n, r0) in (0..nr).step_by(per).enumerate() {
            let r1 = (r0 + per).min(nr);
            spec.push(
                File::new(format!("pt{ci}_{n}"), vec![r1 - r0, px, nt * lv], Elem::F16).with(
                    Fill::Ple(Ple {
                        embed: "embed".into(),
                        proj: "ple.model_proj".into(),
                        norm: "ple.model_norm".into(),
                        tables: tables.clone(),
                        width: ple_w,
                        vocab: v,
                        rows: (r0, r1),
                        nt,
                        layers: layers.clone(),
                        eps: 1e-6,
                    }),
                ),
            );
        }
    }
    Ok(spec)
}
