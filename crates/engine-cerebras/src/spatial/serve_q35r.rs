//! Qwen3.5 on serve8b_region (`Q35R 1`, `codegen::serve8b_region_q35`):
//! qwen35_unit as the grids (one layer) and the regions (`RNL` layers), each
//! layer a gated attention or a Gated DeltaNet by bit `L` of `KINDS` (the
//! checkpoint's `layer_types`). Everything else is serve8b_region's: the
//! handoffs, the routes, the head (the embedding when tied, its norm
//! `1 + w`), the strip, the launch.
//!
//! A unit's PE (`py`, `px`) holds, per layer slot, the tiles serve_q35's
//! units hold on their square grids (serve_q35.rs: the same column split,
//! here over the unit's `PX` columns), stacked `[slot][...]` in each symbol:
//! the MLP and the two norms by the unit's layer, the attention arrays by
//! its attention slot (`NLA` of them), the GDN arrays by its GDN slot
//! (`NLG`). `lkind` [NL][kind, slot] says which (kind 1: GDN). Row `py`
//! holds hidden dims [py Hc, (py + 1) Hc), zero past the model's.

use super::ckpt::{Ckpt, Tensor};
use super::program::ProgramSpec;
use super::serve2d::bf16;
use super::serve8b_region::{Kind, Units};
use super::wafer::rect;
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

/// Qwen3.5's facts the units' tiles need, from the checkpoint's config.
#[derive(Debug, Clone)]
pub struct Q35 {
    /// Layer `l` attends (else a Gated DeltaNet).
    pub attends: Vec<bool>,
    pub hidden: usize,
    pub inter: usize,
    /// GDN: k heads, v heads, k (= v) head dim, conv width.
    pub kh: usize,
    pub vh: usize,
    pub kd: usize,
    pub cw: usize,
    /// Attention: query heads, kv heads, head dim, RoPE's turned dims, theta.
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,
    pub rot: usize,
    pub theta: f64,
}

impl Q35 {
    /// From a Hugging Face Qwen3.5 config (its `text_config`, or the top).
    pub fn of_config(cfg: &serde_json::Value) -> R<Q35> {
        let t = cfg.get("text_config").unwrap_or(cfg);
        let num = |k: &str| -> R<usize> {
            t.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("Qwen3.5 config: no {k}"))
        };
        let attends: Vec<bool> = t
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .ok_or("Qwen3.5 config: no layer_types")?
            .iter()
            .map(|v| v.as_str() == Some("full_attention"))
            .collect();
        let rp = t.get("rope_parameters").unwrap_or(t);
        let theta = rp
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .ok_or("Qwen3.5 config: no rope_theta")?;
        let partial = rp
            .get("partial_rotary_factor")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0);
        let hd = num("head_dim")?;
        let kd = num("linear_key_head_dim")?;
        if num("linear_value_head_dim")? != kd {
            return Err("Qwen3.5: the units take v_dim = k_dim".into());
        }
        Ok(Q35 {
            attends,
            hidden: num("hidden_size")?,
            inter: num("intermediate_size")?,
            kh: num("linear_num_key_heads")?,
            vh: num("linear_num_value_heads")?,
            kd,
            cw: num("linear_conv_kernel_dim")?,
            heads: num("num_attention_heads")?,
            kv_heads: num("num_key_value_heads")?,
            hd,
            rot: (hd as f64 * partial).round() as usize,
            theta,
        })
    }

    /// `KINDS`: bit `l` set where layer `l` attends.
    #[must_use]
    pub fn kinds(&self) -> u64 {
        self.attends
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .fold(0, |k, (l, _)| k | (1 << l))
    }
}

/// The GDN column split and the slots of one unit kind.
#[derive(Debug, Clone, Copy)]
struct Split {
    gk: usize,
    dk: usize,
    dv: usize,
    nla: usize,
    nlg: usize,
}

fn split(lay: &ProgramSpec, kind: Kind) -> R<Split> {
    let k = |n: &str| -> R<usize> {
        lay.knob(n)
            .map(|v| v as usize)
            .ok_or_else(|| format!("serve8b_region Q35R: no {n}"))
    };
    Ok(match kind {
        Kind::Grid => Split {
            gk: k("GK")?,
            dk: k("DK")?,
            dv: k("DV")?,
            nla: 1,
            nlg: 1,
        },
        Kind::Region => Split {
            gk: k("RGK")?,
            dk: k("RDK")?,
            dv: k("RDV")?,
            nla: k("RNLA")?,
            nlg: k("RNLG")?,
        },
        Kind::Attn | Kind::Moe => return Err("serve8b_region Q35R: no serve_gptoss units".into()),
    })
}

fn tensor(ckpt: &Ckpt, name: &str, shape: &[usize]) -> R<Tensor> {
    let t = ckpt.get(name)?;
    t.expect_shape(name, shape)?;
    Ok(t)
}

/// One unit's per-PE arrays, PE-major (`py`, then `px`), `n` words a PE.
struct Tile {
    n: usize,
    data: Vec<f32>,
}

impl Tile {
    fn new(pes: usize, n: usize) -> Tile {
        Tile {
            n,
            data: vec![0.0; pes * n],
        }
    }
    fn at(&mut self, pe: usize, i: usize) -> &mut f32 {
        debug_assert!(i < self.n);
        &mut self.data[pe * self.n + i]
    }
}

/// The weights a build holds in bf16 (one leading pad element a PE).
const W16: &[&str] = &["wm", "wgu", "wd"];
/// The 16-bit KV pool (one leading pad element a PE, zeros).
const KV16: &[&str] = &["kc", "vc"];

fn put(
    dev: &mut dyn Device,
    sym: &str,
    t: &Tile,
    at: (usize, usize),
    (w, h): (usize, usize),
    kv16: bool,
) -> R<()> {
    let r = rect(at.0, at.1, w, h);
    if KV16.contains(&sym) && !kv16 {
        // (an f32 pool: one leading pad word a PE)
        let words: Vec<u32> = t
            .data
            .chunks_exact(t.n)
            .flat_map(|pe| std::iter::once(0).chain(pe.iter().map(|x| x.to_bits())))
            .collect();
        return dev.put(sym, r, t.n + 1, &words);
    }
    if sym == "cw" {
        // (the conv's taps in bf16, no pad)
        let words: Vec<u32> = t.data.iter().map(|&x| u32::from(bf16(x))).collect();
        return dev.put16(sym, r, t.n, &words);
    }
    if W16.contains(&sym) || KV16.contains(&sym) {
        let words: Vec<u32> = t
            .data
            .chunks_exact(t.n)
            .flat_map(|pe| std::iter::once(0).chain(pe.iter().map(|&x| u32::from(bf16(x)))))
            .collect();
        return dev.put16(sym, r, t.n + 1, &words);
    }
    let words: Vec<u32> = t.data.iter().map(|x| x.to_bits()).collect();
    dev.put(sym, r, t.n, &words)
}

/// Every unit's layers (`units` in layer order), as qwen35_unit reads them.
pub(crate) fn place_layers(
    lay: &ProgramSpec,
    units: &Units,
    ckpt: &Ckpt,
    dev: &mut dyn Device,
) -> R<()> {
    let m = Q35::of_config(ckpt.config())?;
    if m.attends.len() != lay.layers() {
        return Err(format!(
            "Qwen3.5 has {} layers; the program {}",
            m.attends.len(),
            lay.layers()
        ));
    }
    if lay.knob("KINDS") != Some(m.kinds()) {
        return Err(format!(
            "the program's KINDS {:?} is not the checkpoint's layer mix {}",
            lay.knob("KINDS"),
            m.kinds()
        ));
    }
    if units.nolayer {
        return Ok(());
    }
    for v in units.units(lay) {
        let layers: Vec<usize> = v.layers.clone().collect();
        let mut sp = split(lay, v.kind)?;
        // (lever D's snake: the layout sizes each region's slots by its own
        // layers, as `q35r_region_n`, at least one of each kind)
        if v.kind == Kind::Region
            && units.yd > 0
            && units.rc == 2
            && units.yd / units.rpy + 2 == units.rr
        {
            let na = layers.iter().filter(|&&l| m.attends[l]).count();
            sp.nla = na.max(1);
            sp.nlg = (layers.len() - na).max(1);
        }
        unit(
            lay,
            &m,
            ckpt,
            v.k,
            &layers,
            (v.x0, v.y0),
            sp,
            units.rot8,
            dev,
        )?;
        tracing::info!(
            unit = v.idx,
            layers = ?layers,
            "spatial wafer: serve8b_region Q35R unit placed"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn unit(
    lay: &ProgramSpec,
    m: &Q35,
    ckpt: &Ckpt,
    k: super::serve8b::Kernel,
    layers: &[usize],
    at: (usize, usize),
    sp: Split,
    rot8: bool,
    dev: &mut dyn Device,
) -> R<()> {
    let (pxn, pyn, hc, ic) = (k.px, k.py, k.hc, k.ic);
    let pes = pxn * pyn;
    let (hid, inter) = (m.hidden, m.inter);
    if pxn * ic != inter {
        return Err(format!(
            "Q35R: {pxn} columns x Ic {ic} is not inter {inter}"
        ));
    }
    if pyn * hc < hid {
        return Err(format!(
            "Q35R: {pyn} rows x Hc {hc} hold less than hidden {hid}"
        ));
    }
    let nl = layers.len();
    let dim = |py: usize, kk: usize| Some(py * hc + kk).filter(|&x| x < hid);
    // The unit's slots: layer li's (kind, slot).
    let mut slots = Vec::with_capacity(nl);
    let (mut na, mut ng) = (0usize, 0usize);
    for &l in layers {
        if m.attends[l] {
            slots.push((0u16, na));
            na += 1;
        } else {
            slots.push((1u16, ng));
            ng += 1;
        }
    }
    if na > sp.nla || ng > sp.nlg {
        return Err(format!(
            "Q35R: layers {layers:?} need {na} attention and {ng} GDN slots; the unit has {} and {}",
            sp.nla, sp.nlg
        ));
    }
    // ---- the MLP and the norms, by layer
    let mut wgu = Tile::new(pes, nl * hc * 2 * ic);
    let mut wd = Tile::new(pes, nl * ic * hc);
    let mut g1 = Tile::new(pes, nl * hc);
    let mut g2 = Tile::new(pes, nl * hc);
    for (li, &l) in layers.iter().enumerate() {
        let n = |s: &str| format!("layers.{l}.{s}");
        let wg = tensor(ckpt, &n("mlp.gate"), &[inter, hid])?;
        let wu = tensor(ckpt, &n("mlp.up"), &[inter, hid])?;
        let wdn = tensor(ckpt, &n("mlp.down"), &[hid, inter])?;
        let n1 = tensor(ckpt, &n("mixer_norm"), &[hid])?;
        let n2 = tensor(ckpt, &n("mlp_norm"), &[hid])?;
        for py in 0..pyn {
            for px in 0..pxn {
                let pe = py * pxn + px;
                for kk in 0..hc {
                    let Some(col) = dim(py, kk) else { continue };
                    for r in 0..ic {
                        let row = px * ic + r;
                        *wgu.at(pe, (li * hc + kk) * 2 * ic + r) = wg.data[row * hid + col];
                        *wgu.at(pe, (li * hc + kk) * 2 * ic + ic + r) = wu.data[row * hid + col];
                        *wd.at(pe, (li * ic + r) * hc + kk) = wdn.data[col * inter + row];
                    }
                    *g1.at(pe, li * hc + kk) = 1.0 + n1.data[col];
                    *g2.at(pe, li * hc + kk) = 1.0 + n2.data[col];
                }
            }
        }
    }
    // ---- attention, by attention slot
    let (dc, gs, qh) = (k.dc, k.gs, lay.qh());
    let (hd, dh, nh1, qo) = (gs * dc, dc / 2, qh + 1, qh * dc);
    if hd != m.hd || pxn / gs != m.kv_heads || m.heads != m.kv_heads * qh {
        return Err(format!(
            "Q35R: attention units of {} kv heads x {hd} (Qh {qh}); the model's {} x {} ({} heads)",
            pxn / gs,
            m.kv_heads,
            m.hd,
            m.heads
        ));
    }
    let rot = m.rot;
    // A head's dim pairs: RoPE's (i, i + rot/2) at frequency i, then the
    // unturned dims in pairs at frequency 0 (serve_q35's).
    let pairs: Vec<(usize, usize, f64)> = (0..rot / 2)
        .map(|i| {
            let f = 1.0 / m.theta.powf(2.0 * i as f64 / rot as f64);
            (i, i + rot / 2, f)
        })
        .chain((0..(hd - rot) / 2).map(|i| (rot + 2 * i, rot + 2 * i + 1, 0.0)))
        .collect();
    let shape = lay.paged().ok_or("Q35R is a paged build")?;
    let kcn = shape.blocks * dc * shape.page_size;
    // The mixer blob a layer (qwen35_unit's `wm`): attention wqa | wqv | wv |
    // wg | wo, GDN wp | wog, each at a multiple of 8 words.
    let r8 = |n: usize| n.div_ceil(8) * 8;
    let (gk0, dk0, dv0) = (sp.gk, sp.dk, sp.dv);
    let vhk0 = m.vh / m.kh;
    let (nv0, npj0) = (vhk0 * dv0, 2 * dk0 + 2 * vhk0 * dv0 + 2 * vhk0);
    let _ = gk0;
    let oqv = r8(hc * nh1 * dh);
    let ov = oqv + r8(hc * nh1 * dh);
    let og = ov + r8(hc * dc);
    let oo = og + r8(hc * qo);
    let mwa = oo + r8(qo * hc);
    let oog = r8(hc * npj0);
    let mwg = oog + r8(nv0 * hc);
    let mw = mwa.max(mwg);
    let mut wm = Tile::new(pes, nl * mw);
    let mut qkn = Tile::new(pes, sp.nla * 2 * nh1 * dh);
    let (units_r, len_r): (&[u32], u32) = if rot8 {
        (&[512, 64, 8, 1], 8)
    } else {
        (&[256, 16, 1], 16)
    };
    let nrot = units_r.len() * len_r as usize * dh;
    let mut rot_c = Tile::new(pes, nrot);
    let mut rot_s = Tile::new(pes, nrot);
    for c in 0..pxn {
        let j = c % gs;
        let prs = &pairs[j * dh..(j + 1) * dh];
        for py in 0..pyn {
            let pe = py * pxn + c;
            for (lv, &u) in units_r.iter().enumerate() {
                for t in 0..len_r {
                    for (kk, p) in prs.iter().enumerate() {
                        let ang = f64::from(t * u) * p.2;
                        let i = (lv * len_r as usize + t as usize) * dh + kk;
                        *rot_c.at(pe, i) = ang.cos() as f32;
                        *rot_s.at(pe, i) = ang.sin() as f32;
                    }
                }
            }
        }
    }
    let sqrt_d = (hd as f64).sqrt() as f32;
    for (li, &l) in layers.iter().enumerate() {
        let (kind, a) = slots[li];
        if kind != 0 {
            continue;
        }
        let n = |s: &str| format!("layers.{l}.attn.{s}");
        let heads = m.heads;
        let qg = tensor(ckpt, &n("qg_proj"), &[2 * heads * hd, hid])?;
        let wk = tensor(ckpt, &n("k_proj"), &[m.kv_heads * hd, hid])?;
        let wv = tensor(ckpt, &n("v_proj"), &[m.kv_heads * hd, hid])?;
        let wo = tensor(ckpt, &n("o_proj"), &[hid, heads * hd])?;
        let qn = tensor(ckpt, &n("q_norm"), &[hd])?;
        let kn = tensor(ckpt, &n("k_norm"), &[hd])?;
        let q_row = |hq: usize, dd: usize| hq * 2 * hd + dd;
        let g_row = |hq: usize, dd: usize| hq * 2 * hd + hd + dd;
        for c in 0..pxn {
            let (g, j) = (c / gs, c % gs);
            let prs = &pairs[j * dh..(j + 1) * dh];
            let dl: Vec<usize> = prs
                .iter()
                .map(|t| t.0)
                .chain(prs.iter().map(|t| t.1))
                .collect();
            let head_rows = |hi: usize| -> Vec<(&Tensor, usize)> {
                if hi < qh {
                    dl.iter().map(|&dd| (&qg, q_row(qh * g + hi, dd))).collect()
                } else {
                    dl.iter().map(|&dd| (&wk, g * hd + dd)).collect()
                }
            };
            let rows: Vec<Vec<(&Tensor, usize)>> = (0..nh1).map(head_rows).collect();
            let norm = |hi: usize, i: usize| -> f32 {
                if hi < qh {
                    (1.0 + qn.data[dl[i]]) / sqrt_d
                } else {
                    1.0 + kn.data[dl[i]]
                }
            };
            let o_cols: Vec<usize> = (0..qh)
                .flat_map(|hi| dl.iter().map(move |&dd| (qh * g + hi) * hd + dd))
                .collect();
            for py in 0..pyn {
                let pe = py * pxn + c;
                for jj in 0..hc {
                    let Some(col) = dim(py, jj) else { continue };
                    let m0 = li * mw;
                    for (hi, hr) in rows.iter().enumerate() {
                        for kk in 0..dh {
                            let i = (jj * nh1 + hi) * dh + kk;
                            let (t, r) = hr[kk];
                            *wm.at(pe, m0 + i) = t.data[r * hid + col];
                            let (t, r) = hr[dh + kk];
                            *wm.at(pe, m0 + oqv + i) = t.data[r * hid + col];
                        }
                    }
                    for (kk, &dd) in dl.iter().enumerate() {
                        *wm.at(pe, m0 + ov + jj * dc + kk) = wv.data[(g * hd + dd) * hid + col];
                        for hi in 0..qh {
                            *wm.at(pe, m0 + og + (jj * qh + hi) * dc + kk) =
                                qg.data[g_row(qh * g + hi, dd) * hid + col];
                        }
                    }
                    for (i, &oc) in o_cols.iter().enumerate() {
                        *wm.at(pe, m0 + oo + i * hc + jj) = wo.data[col * heads * hd + oc];
                    }
                }
                for hi in 0..nh1 {
                    for kk in 0..dh {
                        *qkn.at(pe, ((a * 2) * nh1 + hi) * dh + kk) = norm(hi, kk);
                        *qkn.at(pe, ((a * 2 + 1) * nh1 + hi) * dh + kk) = norm(hi, dh + kk);
                    }
                }
            }
        }
    }
    // ---- the Gated DeltaNet, by GDN slot
    let (gk, dk, dv) = (sp.gk, sp.dk, sp.dv);
    let vhk = m.vh / m.kh;
    let (kd, vd) = (m.kd, m.kd);
    if gk * dk != kd || gk * dv != vd || pxn / gk != m.kh || pyn < kd {
        return Err(format!(
            "Q35R: GDN units of {} k heads x GK {gk} (DK {dk}, DV {dv}) on {pyn} rows; the model's {} x {kd}",
            pxn / gk,
            m.kh
        ));
    }
    let (key, vw) = (m.kh * kd, m.vh * vd);
    let qkvn = 2 * key + vw;
    let nv = vhk * dv;
    let nch = 2 * dk + nv;
    let npj = nch + nv + 2 * vhk;
    let cw = m.cw;
    let mut cwt = Tile::new(pes, sp.nlg * cw * nch);
    let mut gab = Tile::new(pes, sp.nlg * 2 * vhk);
    let mut nw = Tile::new(pes, sp.nlg * nv);
    for (li, &l) in layers.iter().enumerate() {
        let (kind, g) = slots[li];
        if kind != 1 {
            continue;
        }
        let n = |s: &str| format!("layers.{l}.gdn.{s}");
        let in_qkv = tensor(ckpt, &n("in_qkv"), &[qkvn, hid])?;
        let in_z = tensor(ckpt, &n("in_z"), &[vw, hid])?;
        let in_b = tensor(ckpt, &n("in_b"), &[m.vh, hid])?;
        let in_a = tensor(ckpt, &n("in_a"), &[m.vh, hid])?;
        let conv = ckpt.get(&n("conv"))?;
        if conv.data.len() != qkvn * cw {
            return Err(format!(
                "{}: {:?}, not {qkvn} x {cw}",
                n("conv"),
                conv.shape
            ));
        }
        let a_log = tensor(ckpt, &n("a_log"), &[m.vh])?;
        let dt_bias = tensor(ckpt, &n("dt_bias"), &[m.vh])?;
        let norm = tensor(ckpt, &n("norm"), &[vd])?;
        let out_proj = tensor(ckpt, &n("out_proj"), &[hid, vw])?;
        for c in 0..pxn {
            let (head, s) = (c / gk, c % gk);
            let qr: Vec<usize> = (0..dk).map(|i| head * kd + s * dk + i).collect();
            let kr: Vec<usize> = qr.iter().map(|r| key + r).collect();
            let jvs: Vec<usize> = (0..vhk).map(|t| head * vhk + t).collect();
            let vdims: Vec<usize> = jvs
                .iter()
                .flat_map(|&jv| (0..dv).map(move |i| jv * vd + s * dv + i))
                .collect();
            let vr: Vec<usize> = vdims.iter().map(|r| 2 * key + r).collect();
            let conv_rows: Vec<usize> = qr.iter().chain(&kr).chain(&vr).copied().collect();
            let rows: Vec<(&Tensor, usize)> = conv_rows
                .iter()
                .map(|&r| (&in_qkv, r))
                .chain(vdims.iter().map(|&r| (&in_z, r)))
                .chain(jvs.iter().map(|&r| (&in_b, r)))
                .chain(jvs.iter().map(|&r| (&in_a, r)))
                .collect();
            for py in 0..pyn {
                let pe = py * pxn + c;
                for jj in 0..hc {
                    let Some(col) = dim(py, jj) else { continue };
                    let m0 = li * mw;
                    for (kk, (t, r)) in rows.iter().enumerate() {
                        *wm.at(pe, m0 + jj * npj + kk) = t.data[r * hid + col];
                    }
                    for (i, &r) in vdims.iter().enumerate() {
                        *wm.at(pe, m0 + oog + i * hc + jj) = out_proj.data[col * vw + r];
                    }
                }
                for t in 0..cw {
                    for (ch, &r) in conv_rows.iter().enumerate() {
                        *cwt.at(pe, (g * cw + t) * nch + ch) = conv.data[r * cw + t];
                    }
                }
                for (t, &jv) in jvs.iter().enumerate() {
                    *gab.at(pe, g * 2 * vhk + t) = -(f64::from(a_log.data[jv]).exp() as f32);
                    *gab.at(pe, g * 2 * vhk + vhk + t) = dt_bias.data[jv];
                }
                for t in 0..vhk {
                    for i in 0..dv {
                        *nw.at(pe, g * nv + t * dv + i) = norm.data[s * dv + i];
                    }
                }
            }
        }
    }
    let b = lay.launch_lanes();
    let kv16 = lay.param("KV16").or(lay.knob("KV16").map(|v| v as usize)) != Some(0);
    let st = Tile::new(pes, sp.nlg * b * nv);
    let cst = Tile::new(pes, sp.nlg * b * (cw - 1) * nch);
    let pool = Tile::new(pes, sp.nla * kcn);
    for (sym, t) in [
        ("wgu", &wgu),
        ("wd", &wd),
        ("gamma1", &g1),
        ("gamma2", &g2),
        ("wm", &wm),
        ("qkn_w", &qkn),
        ("rot_c", &rot_c),
        ("rot_s", &rot_s),
        ("kc", &pool),
        ("vc", &pool),
        ("cw", &cwt),
        ("gab", &gab),
        ("nw", &nw),
        ("st", &st),
        ("cst", &cst),
    ] {
        put(dev, sym, t, at, (pxn, pyn), kv16)?;
    }
    let lk: Vec<u32> = (0..pes)
        .flat_map(|_| {
            slots
                .iter()
                .flat_map(|&(kind, s)| [u32::from(kind), s as u32])
        })
        .collect();
    dev.put16("lkind", rect(at.0, at.1, pxn, pyn), 2 * nl, &lk)
}

/// A unit's KV slots and the pool's memcpy width for the host's page
/// copies: the unit's attention layers (at least one) and 32-bit cells for an
/// f32 pool on a Q35R build; else every layer, bf16.
pub(crate) fn kv_of(
    lay: &ProgramSpec,
    units: &Units,
    v: &super::serve8b_region::Unit,
) -> (usize, crate::sdk::DataType) {
    use crate::sdk::DataType;
    if !lay.q35r() {
        return (v.layers.len(), DataType::Bits16);
    }
    let kv16 = lay.param("KV16").or(lay.knob("KV16").map(|x| x as usize)) != Some(0);
    let dt = if kv16 {
        DataType::Bits16
    } else {
        DataType::Bits32
    };
    let attends = |l: usize| (lay.knob("KINDS").unwrap_or(0) >> l) & 1 == 1;
    let n = match v.kind {
        Kind::Grid => 1,
        Kind::Region if units.yd > 0 && units.rc == 2 && units.yd / units.rpy + 2 == units.rr => {
            v.layers.clone().filter(|&l| attends(l)).count().max(1)
        }
        Kind::Region => lay.knob("RNLA").unwrap_or(1) as usize,
        Kind::Attn | Kind::Moe => v.layers.len(),
    };
    (n, dt)
}

#[cfg(test)]
mod tests {
    use super::super::program::{ProgramSpec, parse_params};
    use super::super::serve8b_region::Kind;

    /// The layout's `q35r_region_l0` (codegen.rs) as Rust: a region cell's
    /// first layer under lever D's snake.
    fn layout_l0(c: usize, r: usize, rr: usize, rnl: usize) -> usize {
        let k = if r + 2 < rr {
            2 * r + if r.is_multiple_of(2) { 1 - c } else { c }
        } else if r + 2 == rr {
            if c == 0 {
                2 * (rr - 2)
            } else {
                2 * (rr - 2) + 3
            }
        } else if c == 0 {
            2 * (rr - 2) + 1
        } else {
            2 * (rr - 2) + 2
        };
        1 + k * rnl
    }

    /// The 4B and 0.8B builds: every region's first layer as the layout
    /// computes it (its slots and code by kind) is the host's.
    #[test]
    fn region_layers_match_the_layout() {
        for params in [
            "PX:128,PY:128,Hc:20,Dc:8,Gs:32,Ic:72,RPX:256,RPY:234,RHc:11,RDc:4,RGs:64,RIc:36,RNL:3,RGY:18,GY:16,RC:2,RR:5,NCOL:1,YD:702,RTR:5,HR:2560,Qh:4,PS:16,NB:1,E:1,B:1,MC:1,W16:1,NP:3,KV16:0,KVCP:0,PROF:0,TIME:1,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:216,Gx:8,Gy:24,vocab:248320,PASSES:3,MODE:0,OVL:0,VMC:1,CAP:64,iters:1,Q35R:1,KINDS:2290649224,KD:128,GK:8,DK:16,DV:16,RGK:16,RDK:8,RDV:8,VH:2,CW:4,RNLA:1,RNLG:3",
            "PX:128,PY:128,Hc:8,Dc:4,Gs:64,Ic:28,RPX:256,RPY:234,RHc:5,RDc:2,RGs:128,RIc:14,RNL:2,RGY:18,GY:16,RC:2,RR:5,NCOL:3,YD:702,RTR:5,HR:1024,Qh:4,PS:16,NB:1,E:1,B:1,MC:1,W16:1,NP:3,KV16:0,KVCP:0,PROF:0,TIME:1,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:216,Gx:8,Gy:24,vocab:248320,PASSES:3,MODE:0,OVL:0,VMC:1,CAP:64,iters:1,Q35R:1,KINDS:8947848,KD:128,GK:8,DK:16,DV:16,RGK:16,RDK:8,RDV:8,VH:1,CW:4,RNLA:1,RNLG:2",
        ] {
            let p = parse_params(params);
            let lay = ProgramSpec::from_params(ProgramSpec::infer_design(&p).unwrap(), &p).unwrap();
            assert!(lay.q35r());
            let super::super::program::Layers::Units { units, .. } = &lay.layers else {
                panic!("units")
            };
            let mut n = 0;
            for v in units.units(&lay) {
                if v.kind == Kind::Region {
                    let (c, r) = (v.x0 / units.rpx, v.y0 / units.rpy);
                    assert_eq!(
                        layout_l0(c, r, units.rr, units.rnl),
                        v.layers.start,
                        "{c},{r}"
                    );
                    n += 1;
                }
            }
            assert_eq!(n, units.rc * units.rr);
        }
    }
}
