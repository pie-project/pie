//! The unit snake's cut (the Dense family): every unit's weight tiles (`w<u>`: QKV / GDN in,
//! out, gate / up, down a layer, each weight PE's slice), the norm gammas (`gam`), GDN's conv,
//! gate params and norm (`cw`, `gpar`, `gnw`), the q / k norms with q's scale (`qkw`), the RoPE
//! tables (`rot`), the head's vocabulary tiles and final norm (`hw`, `hg`), the embedding
//! (`embed`) and the head's embedding tiles (`hemb`).
//!
//! Column mapping (a weight column c of PX): GDN's k head c / (PX / nk), its DK = hk / (PX / nk)
//! dims, v head c / (PX / nv), DV dims; attention's KV head c / (PX / nkv), Dc = hd / (PX / nkv)
//! dims: the first rd / Dc columns hold Dc / 2 rotated pairs (i, i + rd / 2) side by side, the
//! others Dc unrotated dims; Qh = nq / nkv q heads (with their gates) a column. Where PY does not
//! divide the hidden size, Hc = ceil(H / PY) and the padded dims carry zero weights, gammas and
//! embedding.

use super::{Out, Params, R, f16, first};
use crate::spatial::plan::{LayerKind, Plan};

/// A param of layer `l` by its role: the names the traces give it (Qwen3.5's first).
fn layer<'a>(params: &'a dyn Params, l: usize, names: &[&str]) -> R<super::Param<'a>> {
    first(
        params,
        &names
            .iter()
            .map(|n| format!("layers.{l}.{n}"))
            .collect::<Vec<_>>(),
    )
}

struct Cut {
    px: usize,
    py: usize,
    h: usize,
    hc: usize,
}

impl Cut {
    /// `w` [out][H], `idx` [PX][N] -> per weight PE (r, c) the Hc x N tile, row i = input dim
    /// r Hc + i: appended to `parts[r][c]`.
    fn tile_in(&self, w: &[f32], idx: &[Vec<usize>], parts: &mut [Vec<u16>]) {
        let n = idx[0].len();
        for r in 0..self.py {
            for (c, cols) in idx.iter().enumerate() {
                let dst = &mut parts[r * self.px + c];
                for i in 0..self.hc {
                    let d = r * self.hc + i;
                    for &o in cols.iter().take(n) {
                        dst.push(if d < self.h {
                            f16(w[o * self.h + d])
                        } else {
                            0
                        });
                    }
                }
            }
        }
    }

    /// `w` [H][in] (`width` columns), `idx` [PX][N] the column's inputs -> per weight PE the
    /// N x Hc tile.
    fn tile_out(&self, w: &[f32], width: usize, idx: &[Vec<usize>], parts: &mut [Vec<u16>]) {
        for r in 0..self.py {
            for (c, cols) in idx.iter().enumerate() {
                let dst = &mut parts[r * self.px + c];
                for &k in cols {
                    for i in 0..self.hc {
                        let d = r * self.hc + i;
                        dst.push(if d < self.h { f16(w[d * width + k]) } else { 0 });
                    }
                }
            }
        }
    }
}

pub fn stage(plan: &Plan, params: &dyn Params, out: &Out) -> R<()> {
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
    let nw = |w: Vec<f32>| -> Vec<f32> {
        if s.norm.unit_offset {
            w.into_iter().map(|x| 1.0 + x).collect()
        } else {
            w
        }
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
    // GDN's conv taps: its param's width
    let conv = match kinds.iter().position(|g| *g) {
        Some(l) => layer(params, l, &["gdn.conv"])?.shape[1] as usize,
        None => 4,
    };
    let pat = &kinds[..k];
    let nlg = pat.iter().filter(|g| **g).count();
    let inter = s.mlp.intermediate as usize;
    if !inter.is_multiple_of(px) {
        return Err(format!("dense: the MLP's {inter} over {px} columns"));
    }
    let hc = h.div_ceil(py);
    let ic = inter / px;
    let hp = hc * py;
    let (cpk, cpv, cpa) = (px / nk, px / nv, px / nkv);
    let (dk, dv, dc, qh) = (hk / cpk, hv / cpv, hd / cpa, nq / nkv);
    let nrot = dc / 2;
    let nrc = rd / dc;
    let (hcols, hdc) = (py / 2, 2 * hc);
    let (kd, vd) = (nk * hk, nv * hv);
    let cut = Cut { px, py, h, hc };

    let cols = |f: &dyn Fn(usize) -> Vec<usize>| -> Vec<Vec<usize>> { (0..px).map(f).collect() };
    let qidx = |c: usize| -> Vec<usize> {
        (0..dk)
            .map(|j| (c / cpk) * hk + (c % cpk) * dk + j)
            .collect()
    };
    let vidx = |c: usize| -> Vec<usize> {
        (0..dv)
            .map(|j| (c / cpv) * hv + (c % cpv) * dv + j)
            .collect()
    };
    // a column's head dims: rotated pairs side by side, then plain dims
    let perm = |pa: usize| -> Vec<usize> {
        if pa < nrc {
            let h2 = rd / 2;
            (0..nrot)
                .flat_map(|j| [pa * nrot + j, pa * nrot + j + h2])
                .collect()
        } else {
            (0..dc).map(|j| rd + (pa - nrc) * dc + j).collect()
        }
    };
    let pd: Vec<Vec<usize>> = (0..px).map(|c| perm(c % cpa)).collect();

    let nch = 2 * dk + dv;
    let (gw, cww, gpw, gnww, qkww) = (
        2 * k * hc,
        nlg.max(1) * conv * nch,
        nlg.max(1) * 2,
        nlg.max(1) * dv,
        (k - nlg).max(1) * (qh * dc + dc),
    );
    let mut gam = vec![0u16; nu * py * gw];
    let mut cw = vec![0f32; nu * px * cww];
    let mut gpar = vec![0f32; nu * px * gpw];
    let mut gnw = vec![0f32; nu * px * gnww];
    let mut qkw = vec![0f32; nu * px * qkww];
    let isq = (hd as f64).sqrt() as f32;
    let pad = |w: Vec<f32>| -> Vec<f32> {
        let mut w = w;
        w.resize(hp, 0.0);
        w
    };
    for u in 0..nu {
        let mut parts: Vec<Vec<u16>> = vec![Vec::new(); py * px];
        let (mut g, mut a) = (0usize, 0usize);
        for (l, &gdn) in kinds.iter().enumerate().skip(k * u).take(k) {
            let lk = l % k;
            let ln1 = pad(nw(layer(params, l, &["mixer_norm", "attn_norm"])?.f32()?));
            let ln2 = pad(nw(layer(params, l, &["mlp_norm"])?.f32()?));
            for r in 0..py {
                for i in 0..hc {
                    gam[(u * py + r) * gw + lk * 2 * hc + i] = f16(ln1[r * hc + i]);
                    gam[(u * py + r) * gw + lk * 2 * hc + hc + i] = f16(ln2[r * hc + i]);
                }
            }
            if gdn {
                // GDN: [qkv | z | b | a] in, out
                let mut win = layer(params, l, &["gdn.in_qkvz"])?.f32()?;
                win.extend(layer(params, l, &["gdn.in_ba"])?.f32()?);
                let idx = cols(&|c| {
                    let (q, vv, vh) = (qidx(c), vidx(c), c / cpv);
                    let mut o: Vec<usize> = q.clone();
                    o.extend(q.iter().map(|x| kd + x));
                    o.extend(vv.iter().map(|x| 2 * kd + x));
                    o.extend(vv.iter().map(|x| 2 * kd + vd + x));
                    o.push(2 * kd + 2 * vd + vh);
                    o.push(2 * kd + 2 * vd + nv + vh);
                    o
                });
                cut.tile_in(&win, &idx, &mut parts);
                let op = layer(params, l, &["gdn.out_proj"])?;
                cut.tile_out(&op.f32()?, op.shape[1] as usize, &cols(&vidx), &mut parts);
                let cv = layer(params, l, &["gdn.conv"])?.f32()?;
                let al = layer(params, l, &["gdn.a_log"])?.f32()?;
                let dt = layer(params, l, &["gdn.dt_bias"])?.f32()?;
                let gn = layer(params, l, &["gdn.norm"])?.f32()?;
                for c in 0..px {
                    let (q, vv) = (qidx(c), vidx(c));
                    let ch: Vec<usize> = q
                        .iter()
                        .copied()
                        .chain(q.iter().map(|x| kd + x))
                        .chain(vv.iter().map(|x| 2 * kd + x))
                        .collect();
                    let base = (u * px + c) * cww + g * conv * nch;
                    for t in 0..conv {
                        for (j, &chn) in ch.iter().enumerate() {
                            cw[base + t * nch + j] = cv[chn * conv + t];
                        }
                    }
                    let vh = c / cpv;
                    gpar[(u * px + c) * gpw + 2 * g] = al[vh];
                    gpar[(u * px + c) * gpw + 2 * g + 1] = dt[vh];
                    for j in 0..dv {
                        gnw[(u * px + c) * gnww + g * dv + j] = gn[(c % cpv) * dv + j];
                    }
                }
                g += 1;
            } else {
                let win = layer(params, l, &["attn.qg_proj_k_proj_v_proj", "qkv"])?.f32()?;
                let qrow = (1 + usize::from(gate)) * hd;
                let qr = nq * qrow;
                let idx = cols(&|c| {
                    let kvh = c / cpa;
                    let mut o = Vec::new();
                    for x in 0..qh {
                        o.extend(pd[c].iter().map(|d| (qh * kvh + x) * qrow + d));
                    }
                    if gate {
                        for x in 0..qh {
                            o.extend(pd[c].iter().map(|d| (qh * kvh + x) * 2 * hd + hd + d));
                        }
                    }
                    o.extend(pd[c].iter().map(|d| qr + kvh * hd + d));
                    o.extend(pd[c].iter().map(|d| qr + nkv * hd + kvh * hd + d));
                    o
                });
                cut.tile_in(&win, &idx, &mut parts);
                let op = layer(params, l, &["attn.o_proj", "o"])?;
                let oidx = cols(&|c| {
                    let kvh = c / cpa;
                    (0..qh)
                        .flat_map(|x| pd[c].iter().map(move |d| (qh * kvh + x) * hd + d))
                        .collect()
                });
                cut.tile_out(&op.f32()?, op.shape[1] as usize, &oidx, &mut parts);
                let qn = match layer(params, l, &["attn.q_norm", "q_norm"]) {
                    Ok(q) => nw(q.f32()?),
                    Err(_) => vec![1.0; hd],
                };
                let kn = match layer(params, l, &["attn.k_norm", "k_norm"]) {
                    Ok(q) => nw(q.f32()?),
                    Err(_) => vec![1.0; hd],
                };
                for (c, dims) in pd.iter().enumerate() {
                    let base = (u * px + c) * qkww + a * (qh * dc + dc);
                    for x in 0..qh {
                        for (j, &d) in dims.iter().enumerate() {
                            qkw[base + x * dc + j] = qn[d] / isq;
                        }
                    }
                    for (j, &d) in dims.iter().enumerate() {
                        qkw[base + qh * dc + j] = kn[d];
                    }
                }
                a += 1;
            }
            let gu = layer(params, l, &["mlp.gate_up", "gate_up"])?.f32()?;
            let gi = |c: usize| -> Vec<usize> { (0..ic).map(|j| c * ic + j).collect() };
            let idx = cols(&|c| {
                let mut o = gi(c);
                o.extend(gi(c).iter().map(|x| inter + x));
                o
            });
            cut.tile_in(&gu, &idx, &mut parts);
            let dn = layer(params, l, &["mlp.down", "down"])?;
            cut.tile_out(&dn.f32()?, dn.shape[1] as usize, &cols(&gi), &mut parts);
        }
        let width = parts[0].len();
        let blob: Vec<u16> = parts.concat();
        out.save(&format!("w{u}"), &[py, px, width], &blob)?;
    }
    out.save("gam", &[nu, py, gw], &gam)?;
    out.save("cw", &[nu, px, cww], &cw)?;
    out.save("gpar", &[nu, px, gpw], &gpar)?;
    out.save("gnw", &[nu, px, gnww], &gnw)?;
    out.save("qkw", &[nu, px, qkww], &qkw)?;

    // RoPE tables: a position is RLV digits of RBITS bits
    let inv: Vec<f64> = match &s.rope.freqs {
        // the plan's frequencies as its JSON states them
        Some(f) => f
            .iter()
            .map(|x| x.to_string().parse::<f64>().unwrap_or(f64::from(*x)))
            .collect(),
        None => {
            let theta = f64::from(s.rope.theta);
            (0..rd / 2)
                .map(|i| 1.0 / theta.powf((2 * i) as f64 / rd as f64))
                .collect()
        }
    };
    let rb = 1usize << rbits;
    let mut rot = vec![0f32; px * rlv * rb * 2 * nrot];
    for pc in 0..px {
        for j in 0..nrot {
            let f = if pc % cpa < nrc {
                inv[(pc % cpa) * nrot + j]
            } else {
                0.0
            };
            for lv in 0..rlv {
                for d in 0..rb {
                    let ang = (d * rb.pow((rlv - 1 - lv) as u32)) as f64 * f;
                    let at = (((pc * rlv + lv) * rb + d) * 2) * nrot + j;
                    rot[at] = ang.cos() as f32;
                    rot[at + nrot] = ang.sin() as f32;
                }
            }
        }
    }
    out.save("rot", &[px, rlv * rb * 2 * nrot], &rot)?;

    // the head: rows past the vocabulary are copies of its last row (their logits tie with it
    // and the argmax keeps the real, lower id)
    let wh = match params.get("head") {
        Some(p) if !s.head.tied => p,
        _ => first(params, &["embed".into()])?,
    }
    .f32()?;
    let mut hw = Vec::with_capacity(hr * hcols * hdc * vr);
    for big in 0..hr {
        let row = |vrow: usize| -> usize { (big * vr + vrow).min(v - 1) };
        for hcol in 0..hcols {
            for kk in 0..hdc {
                let d = hcol * hdc + kk;
                for vrow in 0..vr {
                    let src = if big * vr + vrow < v {
                        big * vr + vrow
                    } else {
                        row(vrow)
                    };
                    hw.push(if d < h { f16(wh[src * h + d]) } else { 0 });
                }
            }
        }
    }
    out.save("hw", &[hr, hcols, hdc * vr], &hw)?;
    let hg = pad(nw(first(params, &["final_norm".into()])?.f32()?));
    out.save("hg", &[hcols, hdc], &hg)?;
    let eb = first(params, &["embed".into()])?.bf16_bits()?;
    out.save("embed", &[v, h], &eb)?;
    let mut he = vec![0u16; hr * hcols * vr * hdc];
    for big in 0..hr {
        for vrow in 0..vr {
            let id = big * vr + vrow;
            if id >= v {
                continue;
            }
            for d in 0..h {
                let (hcol, kk) = (d / hdc, d % hdc);
                he[((big * hcols + hcol) * vr + vrow) * hdc + kk] = eb[id * h + d];
            }
        }
    }
    out.save("hemb", &[hr, hcols, vr * hdc], &he)?;
    Ok(())
}
