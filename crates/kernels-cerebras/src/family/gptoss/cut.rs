//! The MoE halves' cut (gpt-oss's: MXFP4 experts, bf16 elsewhere; a model with 16-bit experts, such
//! as Qwen3 MoE, the same files: absent biases, sinks and scales stay zero, `qb` holds the q / k
//! norms' weights where the heads are normed); unit u holds layers K u .. K u + K - 1:
//! - `wa<u>` (f16, [PY][AX][K (Hc NQA + QO Hc)]): attention, per layer the qkv tile then o's;
//! - `wx<u>` (E5M2 pairs, [PY][MX][K NEX XE]; 16-bit experts: f16, an expert's gate|up [Hc][2 Ic]
//!   then its down [Ic][Hc]): the experts;
//! - `wr<u>` (f32, [PY][MX][K Hc]): the router rows (output c mod NEX on MoE column c);
//! - `gam<u>`, `bo<u>`, `bd<u>` (f32, the CX's), `qb<u>`, `snk<u>` (the attention CY's), `rb<u>`,
//!   `gb<u>`, `ps<u>` (the MoE CY's: 2^Pg, 2^Pd);
//! - `rot` (YaRN's tables, the attention factor in level 0), `hw` (f16 of 2^15 lm_head,
//!   [PY][HVC][dim][vocab row]), `hg`, `hemb` ([PY][HVC][vocab row][dim], bf16).
//!
//! Column mapping: attention column c = kv head c / SWA, j = c % SWA: rotated pairs
//! i = j NROT + p, p < NROT, dims (i, i + HD / 2) side by side (q: per head of the kv group, its
//! pairs; k; v the same dims); MoE column c: the intermediate dims c Ic .. c Ic + Ic; weight row
//! r: hidden dims r Hc .. r Hc + Hc.

use crate::plan::Plan;
use crate::tiles::spec::{
    Bank, Elem, ExpertScales, Experts, File, Fill, Gather, Grid, Head, Map, Part, Rope, Spec, range,
};

type R<T> = Result<T, String>;

pub fn tiles(plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
    let (s, p) = (&plan.shape, &plan.partition);
    let moe = s.moe.ok_or("gpt-oss: no experts")?;
    let ax = p.attn_cols.ok_or("gpt-oss plan: no attention columns")? as usize;
    let (py, mx) = (p.rows as usize, p.cols as usize - ax);
    let hc = p.row_dims[0].dims as usize;
    let (h, v, nl) = (s.hidden as usize, s.vocab as usize, s.layers.len());
    let (nu, nex) = (p.units as usize, moe.experts as usize);
    let k = nl / nu;
    let (nh, nkv, hd) = (
        s.attn.q_heads as usize,
        s.attn.kv_heads as usize,
        s.attn.head_dim as usize,
    );
    let inter = moe.expert_intermediate as usize;
    let swa = p.segments.scores as usize;
    let (qh, dc) = (nh / nkv, hd / swa);
    let nrot = dc / 2;
    let ic = inter / mx;
    if h != py * hc
        || ax != nkv * swa
        || inter != mx * ic
        || k * nu != nl
        || !mx.is_multiple_of(nex)
    {
        return Err(format!(
            "gpt-oss: {h} hidden on {py} rows of {hc}, {ax} attention columns, {inter} over {mx}"
        ));
    }
    let (nqa, qo) = (qh * dc + 2 * dc, qh * dc);
    let lwa = hc * nqa + qo * hc;
    let cdims = |j: usize| -> Vec<i64> {
        (0..nrot)
            .flat_map(|q| {
                let i = j * nrot + q;
                [i as i64, (i + hd / 2) as i64]
            })
            .collect()
    };
    let qrows = |c: usize| -> Vec<i64> {
        let (kv, j) = (c / swa, c % swa);
        (0..qh)
            .flat_map(|x| {
                cdims(j)
                    .into_iter()
                    .map(move |d| ((kv * qh + x) * hd) as i64 + d)
            })
            .collect()
    };
    let qkv = |c: usize| -> Vec<i64> {
        let (kv, j) = (c / swa, c % swa);
        let krows: Vec<i64> = cdims(j).iter().map(|d| (kv * hd) as i64 + d).collect();
        let mut o = qrows(c);
        o.extend(krows.iter().map(|d| (nh * hd) as i64 + d));
        o.extend(krows.iter().map(|d| ((nh + nkv) * hd) as i64 + d));
        o
    };
    let ln = |l: usize, n: &str| format!("layers.{l}.{n}");
    let need = |n: String| -> R<String> {
        shapes(&n)
            .map(|_| n.clone())
            .ok_or_else(|| format!("gpt-oss: no param {n}"))
    };
    // a layer's param by its names (gpt-oss's import's, then the plain layout's)
    let one = |l: usize, names: &[&str]| -> R<String> {
        names
            .iter()
            .map(|n| ln(l, n))
            .find(|n| shapes(n).is_some())
            .ok_or_else(|| format!("gpt-oss: no param layers.{l}.{}", names.join(" | ")))
    };
    // q / k / v rows: one stacked param, or the three stacked here
    let qkv_src = |l: usize| -> R<Vec<String>> {
        match one(l, &["attn.qkv", "qkv"]) {
            Ok(n) => Ok(vec![n]),
            Err(_) => ["q", "k", "v"].iter().map(|n| need(ln(l, n))).collect(),
        }
    };
    let (a, quant) = (&s.attn, moe.quantized);
    let rows: Vec<(usize, usize)> = (0..py).map(|r| (r * hc, hc)).collect();
    let id = |at: usize, src: String, idx: Vec<i64>| {
        Fill::Gather(Gather {
            at,
            src,
            idx,
            map: Map::Id,
        })
    };
    let mut spec = Spec::default();
    for u in 0..nu {
        let layers: Vec<usize> = (k * u..k * u + k).collect();
        let (mut wa, mut wr, mut wx16) = (Vec::new(), Vec::new(), Vec::new());
        let mut gam = File::new(format!("gam{u}"), vec![py, 2 * k * hc], Elem::F32);
        let mut bo = File::new(format!("bo{u}"), vec![py, k * hc], Elem::F32);
        let mut bd = File::new(format!("bd{u}"), vec![py, k * nex * hc], Elem::F32);
        let mut qb = File::new(format!("qb{u}"), vec![ax, k * nqa], Elem::F32);
        let mut snk = File::new(format!("snk{u}"), vec![ax, k * qh], Elem::F32);
        let mut rb = File::new(format!("rb{u}"), vec![mx, k], Elem::F32);
        let mut gb = File::new(format!("gb{u}"), vec![mx, k * nex * 2 * ic], Elem::F32);
        let mut banks = Vec::new();
        for (l, &ly) in layers.iter().enumerate() {
            wa.push(Part::In {
                src: qkv_src(ly)?,
                outs: (0..ax).map(qkv).collect(),
            });
            wa.push(Part::Out {
                src: vec![one(ly, &["attn.o_proj", "o"])?],
                ins: (0..ax).map(qrows).collect(),
            });
            wr.push(Part::In {
                src: vec![one(ly, &["mlp.router", "router"])?],
                outs: (0..mx).map(|c| vec![(c % nex) as i64]).collect(),
            });
            for r in 0..py {
                gam.fills.push(id(
                    r * 2 * k * hc + 2 * l * hc,
                    ln(ly, "attn_norm"),
                    range(r * hc, hc),
                ));
                gam.fills.push(id(
                    r * 2 * k * hc + (2 * l + 1) * hc,
                    ln(ly, "mlp_norm"),
                    range(r * hc, hc),
                ));
                if a.o_bias {
                    bo.fills.push(id(
                        r * k * hc + l * hc,
                        ln(ly, "attn.o_bias"),
                        range(r * hc, hc),
                    ));
                }
                for e in (0..nex).filter(|_| moe.expert_bias) {
                    bd.fills.push(id(
                        r * k * nex * hc + (l * nex + e) * hc,
                        ln(ly, "mlp.down_bias"),
                        range(e * h + r * hc, hc),
                    ));
                }
            }
            for c in 0..ax {
                if a.bias {
                    qb.fills
                        .push(id(c * k * nqa + l * nqa, ln(ly, "attn.qkv_bias"), qkv(c)));
                }
                if a.qk_norm {
                    // the norms' weights: q's, a head of the kv group each, then k's [Qh Dc | Dc]
                    let (qn, kn) = (
                        one(ly, &["attn.q_norm", "q_norm"])?,
                        one(ly, &["attn.k_norm", "k_norm"])?,
                    );
                    for x in 0..qh {
                        qb.fills.push(id(
                            c * k * nqa + l * nqa + x * dc,
                            qn.clone(),
                            cdims(c % swa),
                        ));
                    }
                    qb.fills
                        .push(id(c * k * nqa + l * nqa + qh * dc, kn, cdims(c % swa)));
                }
                if a.sinks {
                    let kv = c / swa;
                    snk.fills.push(id(
                        c * k * qh + l * qh,
                        ln(ly, "attn.sinks"),
                        range(kv * qh, qh),
                    ));
                }
            }
            for c in 0..mx {
                if moe.router_bias {
                    rb.fills.push(id(
                        c * k + l,
                        ln(ly, "mlp.router_bias"),
                        vec![(c % nex) as i64],
                    ));
                }
                for e in (0..nex).filter(|_| moe.expert_bias) {
                    let at = c * k * nex * 2 * ic + (l * nex + e) * 2 * ic;
                    let mut idx = range(e * 2 * inter + c * ic, ic);
                    idx.extend(range(e * 2 * inter + inter + c * ic, ic));
                    gb.fills.push(id(at, ln(ly, "mlp.gate_up_bias"), idx));
                }
            }
            if quant {
                banks.push(Bank {
                    gate_up: need(ln(ly, "mlp.gate_up"))?,
                    gate_up_scales: need(ln(ly, "mlp.gate_up.scales"))?,
                    down: need(ln(ly, "mlp.down"))?,
                    down_scales: need(ln(ly, "mlp.down.scales"))?,
                });
            } else {
                // 16-bit experts ([experts][rows][cols] banks): each expert's gate|up rows of the
                // column's dims (gate's, then up's), then its down columns of them
                // ([experts][gate rows | up rows][hidden], [experts][hidden][inter])
                let (gate_up, down) = (
                    one(ly, &["mlp.gate_up", "gate_up"])?,
                    one(ly, &["mlp.down", "down"])?,
                );
                for e in 0..nex {
                    wx16.push(Part::In {
                        src: vec![gate_up.clone()],
                        outs: (0..mx)
                            .map(|c| {
                                let mut o = range(e * 2 * inter + c * ic, ic);
                                o.extend(range(e * 2 * inter + inter + c * ic, ic));
                                o
                            })
                            .collect(),
                    });
                    wx16.push(Part::Out {
                        src: vec![down.clone()],
                        ins: (0..mx).map(|c| range(e * h * inter + c * ic, ic)).collect(),
                    });
                }
            }
        }
        let (xg, xd) = if quant {
            ((hc / 2) * 2 * ic, (ic / 2) * hc)
        } else {
            (hc * 2 * ic, ic * hc)
        };
        spec.push(
            File::new(format!("wa{u}"), vec![py, ax, k * lwa], Elem::F16).with(Fill::Grid(Grid {
                hidden: h,
                rows: rows.clone(),
                parts: wa,
            })),
        );
        let wx = vec![py, mx, k * nex * (xg + xd)];
        spec.push(if quant {
            File::new(format!("wx{u}"), wx, Elem::U16).with(Fill::Experts(Experts {
                banks: banks.clone(),
                experts: nex,
                hidden: h,
                inter,
                rows: py,
                cols: mx,
            }))
        } else {
            File::new(format!("wx{u}"), wx, Elem::F16).with(Fill::Grid(Grid {
                hidden: h,
                rows: rows.clone(),
                parts: wx16,
            }))
        });
        spec.push(
            File::new(format!("wr{u}"), vec![py, mx, k * hc], Elem::F32).with(Fill::Grid(Grid {
                hidden: h,
                rows: rows.clone(),
                parts: wr,
            })),
        );
        for f in [gam, bo, bd, qb, snk, rb, gb] {
            spec.push(f);
        }
        let ps = File::new(format!("ps{u}"), vec![mx, k * nex * 2], Elem::F32);
        spec.push(if quant {
            ps.with(Fill::ExpertScales(ExpertScales {
                banks,
                experts: nex,
                cols: mx,
            }))
        } else {
            ps
        });
    }

    // RoPE tables: level lv, digit d: angle d 8^(levels - 1 - lv) f; YaRN's attention factor in
    // level 0
    let inv = s.rope.inv_freq(s.attn.head_dim);
    let (levels, radix) = (
        plan.contract.rope_levels as usize,
        1usize << plan.contract.rope_bits,
    );
    spec.push(
        File::new("rot", vec![ax, levels * radix * 2 * nrot], Elem::F32).with(Fill::Rope(Rope {
            freqs: (0..ax)
                .map(|c| vec![(0..nrot).map(|q| inv[(c % swa) * nrot + q]).collect()])
                .collect(),
            levels,
            radix,
            scale0: s.rope.scale(),
        })),
    );

    // the head: rows past the vocabulary copy its last row
    let (hvc, vr) = (p.head.cols as usize, p.head.vocab_rows as usize);
    let dims: Vec<Vec<i64>> = (0..py).map(|r| range(r * hc, hc)).collect();
    let head = if s.head.tied { "embed" } else { "head" };
    let tiles = |src: &str, row_major: bool, scale: f32| Head {
        src: src.into(),
        vocab: v,
        groups: hvc,
        per: vr,
        dims: dims.clone(),
        dims_major: true,
        row_major,
        copy_last: true,
        scale,
    };
    spec.push(
        File::new("hw", vec![py, hvc, hc * vr], Elem::F16)
            .with(Fill::Head(tiles(head, false, 32768.0))),
    );
    spec.push(File::new("hg", vec![py, hc], Elem::F32).with(Fill::Copy {
        at: 0,
        src: "final_norm".into(),
        map: Map::Id,
    }));
    spec.push(
        File::new("hemb", vec![py, hvc, vr * hc], Elem::Bf16)
            .with(Fill::Head(tiles("embed", true, 1.0))),
    );
    Ok(spec)
}
