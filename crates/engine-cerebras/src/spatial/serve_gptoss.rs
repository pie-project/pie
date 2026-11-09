//! The serve_gptoss wafer program (`spatial/csl/serve_gptoss`): gpt-oss's
//! layers as units in slots, and its host side. Per layer an attention unit
//! (gptoss_attn, `PX` x `AY`, `AHc` dims a row) over a MoE unit (gptoss_moe,
//! `PX` x `MY`, `MHc` dims a row) in a slot of `GW` x `GH`; the head and the
//! embedding strip east of the slots.
//!
//! Layer order ([`Slots::units`]): slot row 0 from c = GW - 1 west to 0,
//! then the rows below serpentine by columns (column 0 down, 1 up, ...).
//! The handoffs are serve8b_region's pieces ([`Tables`]): attention -> MoE
//! is re-chunk A down the slot; MoE -> the next attention unit chains a dest
//! row's pieces in a MoE column and legs them down, up, east or west to the
//! row's assembling PE, which multicasts the row. This is the bare host's
//! `payload/serve_gptoss/gptoss_host.Tables`, word for word.

use super::ckpt::Ckpt;
use super::device::Device;
use super::moe_place::{MoeGrid, MoeLayer, MoeTiles, pack_moe};
use super::paged::{NONE, Shape};
use super::serve8b::Kernel;
use super::serve8b_region::{Kind, Piece, Tables, Unit, route_words, route_words_arf};
use super::wafer::rect;
use crate::sdk::DataType;

type R<T> = std::result::Result<T, String>;

const FWD: u32 = 0xffff;
const D_RAMP: u32 = 0;
const D_E: u32 = 3;
/// Row multicasts by the receiving unit's parity.
const C_R: [u32; 2] = [11, 12];
/// paths.csl: S (the last unit's chunk triples -> head), E (strip -> layer 0).
const C_S: u32 = 20;
const C_E: u32 = 13;
/// MoE -> attention by the dest layer's parity: legs, hops.
const M_LEG: [u32; 2] = [4, 17];
const M_HOP: [[u32; 2]; 2] = [[14, 15], [18, 19]];

/// serve_gptoss's slots and unit shapes (its layout params).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slots {
    pub px: usize,
    pub ay: usize,
    pub ahc: usize,
    pub my: usize,
    pub mhc: usize,
    pub gw: usize,
    pub gh: usize,
    pub nl: usize,
}

impl Slots {
    /// The slots (c, d) in layer order.
    pub fn order(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = (0..self.gw).rev().map(|c| (c, 0)).collect();
        for c in 0..self.gw {
            if c % 2 == 0 {
                out.extend((1..self.gh).map(|d| (c, d)));
            } else {
                out.extend((1..self.gh).rev().map(|d| (c, d)));
            }
        }
        out.truncate(self.nl);
        out
    }

    /// The units in layer order: layer l's attention unit (idx 2 l), then its
    /// MoE unit (2 l + 1).
    pub fn units(&self) -> Vec<Unit> {
        let sy = self.ay + self.my;
        let kernel = |py, hc| Kernel {
            px: self.px,
            py,
            hc,
            dc: 0,
            gs: 0,
            ic: 0,
        };
        let mut out = Vec::new();
        for (l, (c, d)) in self.order().into_iter().enumerate() {
            for (kind, y0, k) in [
                (Kind::Attn, d * sy, kernel(self.ay, self.ahc)),
                (Kind::Moe, d * sy + self.ay, kernel(self.my, self.mhc)),
            ] {
                out.push(Unit {
                    kind,
                    idx: out.len(),
                    x0: c * self.px,
                    y0,
                    k,
                    layers: l..l + 1,
                });
            }
        }
        out
    }
}

/// Every unit PE's handoff table (`rtab`, [`Tables::table`]): `rtr` routes
/// and `rti` pieces in a PE.
pub fn tables(s: &Slots, lanes: usize, rtr: usize, rti: usize) -> R<Tables> {
    let units = s.units();
    let mut t = Tables::empty(units.clone(), lanes, rtr);
    t.rti = rti;
    let (u0, ul) = (&units[0], &units[units.len() - 1]);
    // E: layer 0's row py from the east.
    for py in 0..u0.k.py {
        t.row_mcast(C_E, u0, py, u0.x0 + u0.k.px - 1, D_E)?;
        for x in u0.x0..u0.x0 + u0.k.px {
            t.recv(x, u0.y0 + py, C_E, 0, u0.k.hc as u32)?;
        }
    }
    // S: the last unit's rows 3k..3k+2 chained down its east column, east
    // from row 3k + 2.
    let x = ul.x0 + ul.k.px - 1;
    let hop = M_HOP[0];
    let hc = ul.k.hc as u32;
    for k in 0..ul.k.py / 3 {
        for i in 0..3 {
            let y = ul.y0 + 3 * k + i;
            let color = if i == 2 { C_S } else { hop[i % 2] };
            if i > 0 {
                t.set_fwd_in(x, y, hop[(i - 1) % 2])?;
                t.send(x, y, color, FWD, t.lanes * hc * i as u32)?;
            }
            t.send(x, y, color, 0, hc)?;
            if i < 2 {
                t.leg(color, &[(x, y), (x, y + 1)])?;
            } else {
                t.route(x, y, C_S, D_RAMP, D_E, D_E)?;
            }
        }
    }
    for w in units.windows(2) {
        handoff(&mut t, &w[0], &w[1])?;
    }
    Ok(t)
}

fn handoff(t: &mut Tables, a: &Unit, b: &Unit) -> R<()> {
    let r = C_R[b.idx % 2];
    if a.kind == Kind::Attn {
        if b.x0 != a.x0 || b.y0 != a.y0 + a.k.py {
            return Err(format!("unit {}: its MoE unit is not below it", a.idx));
        }
        return t.rechunk_a(a, b, r);
    }
    let p = b.layers.start % 2;
    let (leg, hop) = (M_LEG[p], M_HOP[p]);
    for g in 0..b.k.py {
        let pieces = Tables::pieces(a, b, g);
        if pieces.is_empty() {
            continue;
        }
        let srcs: Vec<Piece> = pieces
            .iter()
            .map(|&(row, so, d, n)| (a.y0 + row, so, d, n))
            .collect();
        let (top, bottom) = (srcs[0].0, srcs[srcs.len() - 1].0);
        let yg = b.y0 + g;
        let dest = if b.x0 == a.x0 && b.y0 == a.y0 + a.k.py {
            let col = a.x0 + g;
            let dest = (col, yg);
            t.chain(col, &srcs, hop, leg, &[(col, bottom), dest], dest, true)?;
            dest
        } else if b.x0 == a.x0 && b.y0 < a.y0 {
            let col = a.x0 + g;
            let dest = (col, yg);
            t.chain(col, &srcs, hop, leg, &[(col, top), dest], dest, false)?;
            dest
        } else if b.x0 == a.x0 + a.k.px && b.y0 + b.k.py == a.y0 {
            let col = a.x0 + a.k.px - b.k.py + g;
            let dest = (b.x0, yg);
            t.chain(
                col,
                &srcs,
                hop,
                leg,
                &[(col, top), (col, yg), dest],
                dest,
                false,
            )?;
            dest
        } else if b.x0 + a.k.px == a.x0 && b.y0 + b.k.py == a.y0 {
            let col = a.x0 + b.k.py - 1 - g;
            let dest = (b.x0 + b.k.px - 1, yg);
            t.chain(
                col,
                &srcs,
                hop,
                leg,
                &[(col, top), (col, yg), dest],
                dest,
                false,
            )?;
            dest
        } else {
            return Err(format!(
                "no handoff from unit {} at {:?} to unit {} at {:?}",
                a.idx,
                (a.x0, a.y0),
                b.idx,
                (b.x0, b.y0)
            ));
        };
        t.set_relay(dest.0, dest.1, r)?;
        t.row_mcast(r, b, g, dest.0, D_RAMP)?;
        for x in b.x0..b.x0 + b.k.px {
            if x != dest.0 {
                t.recv(x, yg, r, 0, b.k.hc as u32)?;
            }
        }
    }
    Ok(())
}

/// serve_gptoss's params beyond the MoE grid's kernel (`Layers::Gptoss`):
/// the slots, the units' splits and handoff capacities, the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt {
    pub slots: Slots,
    /// `Dc`, `Gs`: an attention column's dims of its kv head, columns a kv
    /// head; `Ic`: a MoE column's intermediate dims.
    pub dc: usize,
    pub gs: usize,
    pub ic: usize,
    /// `NE`, `TK`: experts and the routed ones a token.
    pub ne: usize,
    pub tk: usize,
    /// `AGY`, `MGY`: the units' row all-reduce groups.
    pub agy: usize,
    pub mgy: usize,
    /// `SLW`: layer `l` attends over the window where bit `l` is set;
    /// `WIN`: the window.
    pub slw: u64,
    pub win: usize,
    pub rtr: usize,
    pub rti: usize,
    pub hroute: bool,
    /// `KV16`: the pool in bf16 (else f32).
    pub kv16: bool,
    /// `QKV1` (the levered attention units, codegen::gptoss_attn_lev): one
    /// QKV matvec, `wqa` = [Hc][q a | q v | v]; `wqv` and `wv` are 2-word
    /// dummies the host does not write.
    pub qkv1: bool,
    /// `ARF` (the levered attention units with GRD 1): their Y (and with bit
    /// 2 their X) full all-reduces as FIFO streamed halves; the host writes
    /// those units' ARF route words (12 a PE, 18 under ARF 3). The MoE units
    /// keep the grouped all-reduce.
    pub arf: usize,
}

impl Gpt {
    /// The attention units' kernel.
    #[must_use]
    pub fn attn(&self) -> Kernel {
        Kernel {
            px: self.slots.px,
            py: self.slots.ay,
            hc: self.slots.ahc,
            dc: self.dc,
            gs: self.gs,
            ic: 0,
        }
    }

    /// Whether layer `l` attends over the window.
    #[must_use]
    pub fn windowed(&self, l: usize) -> bool {
        l < 64 && self.slw >> l & 1 == 1
    }
}

// ----------------------------------------------------------------- placement
//
// The bare host's (`gptoss_host.WaferModel._attn`, `._moe`, `._head`,
// `._embed`), word for word, from a Hugging Face directory by gpt-oss's
// names (a file staged in pieces read as one).

/// f32 bits as bf16 (round to nearest even).
fn bf16(x: f32) -> u32 {
    u32::from(super::ckpt::bf16_bits(x))
}

fn widen(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// `per_pe` elements a PE of unit `u`, built by `pe(py, px, out)`.
fn tiles(u: &Unit, per_pe: usize, mut pe: impl FnMut(usize, usize, &mut Vec<u32>)) -> R<Vec<u32>> {
    let mut out = Vec::with_capacity(u.k.px * u.k.py * per_pe);
    for py in 0..u.k.py {
        for px in 0..u.k.px {
            let n = out.len();
            pe(py, px, &mut out);
            if out.len() != n + per_pe {
                return Err(format!(
                    "unit {}: a PE's tile is {} words, not {per_pe}",
                    u.idx,
                    out.len() - n
                ));
            }
        }
    }
    Ok(out)
}

fn put32(
    dev: &mut dyn Device,
    sym: &str,
    u: &Unit,
    n: usize,
    pe: impl FnMut(usize, usize, &mut Vec<u32>),
) -> R<()> {
    let w = tiles(u, n, pe)?;
    dev.put(sym, rect(u.x0, u.y0, u.k.px, u.k.py), n, &w)
}

/// bf16 bits with one leading pad element (`_put16`).
fn put16(
    dev: &mut dyn Device,
    sym: &str,
    u: &Unit,
    n: usize,
    mut pe: impl FnMut(usize, usize, &mut Vec<u32>),
) -> R<()> {
    let w = tiles(u, n + 1, |py, px, out| {
        out.push(0);
        pe(py, px, out);
    })?;
    dev.put16(sym, rect(u.x0, u.y0, u.k.px, u.k.py), n + 1, &w)
}

/// YaRN's frequencies and cos/sin scale from the checkpoint's config
/// (gptoss_ref.yarn_inv_freq: Hugging Face's, in f32, the power correctly
/// rounded).
pub fn yarn_inv_freq(cfg: &serde_json::Value) -> R<(Vec<f32>, f32)> {
    let rs = &cfg["rope_scaling"];
    let num = |v: &serde_json::Value, k: &str| {
        v[k].as_f64()
            .ok_or_else(|| format!("the checkpoint config has no {k}"))
    };
    let base = num(cfg, "rope_theta")?;
    let dim = cfg["head_dim"]
        .as_u64()
        .ok_or("the checkpoint config has no head_dim")? as usize;
    let (factor, bf, bs) = (
        num(rs, "factor")?,
        num(rs, "beta_fast")?,
        num(rs, "beta_slow")?,
    );
    let omax = num(rs, "original_max_position_embeddings")?;
    let corr = |rot: f64| {
        (dim as f64 * (omax / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln())
    };
    let (mut low, mut high) = (corr(bf), corr(bs));
    if rs["truncate"].as_bool().unwrap_or(true) {
        (low, high) = (low.floor(), high.ceil());
    }
    (low, high) = (low.max(0.0), high.min((dim - 1) as f64));
    if low == high {
        high += 0.001;
    }
    let span = (high - low) as f32;
    let inv = (0..dim / 2)
        .map(|i| {
            let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
            // base^(2i/d) correctly rounded: numpy's float32 power (SVML
            // on AVX-512 hosts) is not, and differs by host.
            let e = (2 * i) as f32 / dim as f32;
            let pos = f64::from(base as f32).powf(f64::from(e)) as f32;
            let extra = 1.0f32 / pos;
            let inter = 1.0f32 / (factor as f32 * pos);
            let ef = 1.0f32 - ramp;
            inter * (1.0 - ef) + extra * ef
        })
        .collect();
    let att = rs["attention_factor"]
        .as_f64()
        .filter(|&a| a != 0.0)
        .unwrap_or(0.1 * factor.ln() + 1.0);
    Ok((inv, att as f32))
}

fn f32s(ckpt: &Ckpt, leaf: &str, n: usize) -> R<Vec<f32>> {
    let t = ckpt.hf(leaf)?;
    if t.data.len() != n {
        return Err(format!(
            "{leaf} is {:?}; the program wants {n} values",
            t.shape
        ));
    }
    Ok(t.data)
}

fn bits(ckpt: &Ckpt, leaf: &str, shape: &[usize]) -> R<Vec<u16>> {
    let (have, b) = ckpt.hf_bits(leaf)?;
    if have != shape {
        return Err(format!("{leaf} is {have:?}; the program wants {shape:?}"));
    }
    Ok(b)
}

/// Layer `u.layers.start`'s attention unit (`_attn`): per column `px` (kv
/// head `px / Gs`, slice `j = px % Gs`) the head dims `j DH ..` and `D/2 +
/// j DH ..` (DH = Dc / 2) of its `Qh` query heads and the kv head (`wqa`
/// the first halves, `wqv` the second, `[Hc][Qh + 1][DH]`), V's `[Hc][Dc]`,
/// O's `[Qh Dc][Hc]`, row `py` the hidden dims `py Hc ..`; softmax scale
/// 1/sqrt(D) folded into Wq and bq, YaRN's mscale into RoPE level 0.
pub fn place_attn(
    gpt: &Gpt,
    hidden: usize,
    u: &Unit,
    ckpt: &Ckpt,
    yarn: &(Vec<f32>, f32),
    lanes: usize,
    dev: &mut dyn Device,
) -> R<()> {
    let l = u.layers.start;
    let p = format!("model.layers.{l}.self_attn.");
    let (px_n, hc) = (u.k.px, u.k.hc);
    let (dc, gs) = (gpt.dc, gpt.gs);
    let d = gs * dc;
    let dh = dc / 2;
    let nkv = px_n / gs;
    let qh = {
        let (shape, _) = ckpt.hf_bits(&format!("{p}sinks"))?;
        shape[0] / nkv
    };
    let nh = qh + 1;
    let sm = (d as f32).sqrt().recip();
    if sm != 2f32.powi(sm.log2().round() as i32) {
        return Err(format!("1/sqrt({d}) does not fold into bf16 Wq exactly"));
    }
    let nq = nkv * qh * d;
    let wq: Vec<u16> = bits(ckpt, &format!("{p}q_proj.weight"), &[nq, hidden])?
        .into_iter()
        .map(|b| bf16(widen(b) * sm) as u16)
        .collect();
    let wk = bits(ckpt, &format!("{p}k_proj.weight"), &[nkv * d, hidden])?;
    let wv = bits(ckpt, &format!("{p}v_proj.weight"), &[nkv * d, hidden])?;
    let wo = bits(ckpt, &format!("{p}o_proj.weight"), &[hidden, nq])?;
    let bq: Vec<f32> = f32s(ckpt, &format!("{p}q_proj.bias"), nq)?
        .into_iter()
        .map(|x| x * sm)
        .collect();
    let bk = f32s(ckpt, &format!("{p}k_proj.bias"), nkv * d)?;
    let bv = f32s(ckpt, &format!("{p}v_proj.bias"), nkv * d)?;
    let sinks = f32s(ckpt, &format!("{p}sinks"), nkv * qh)?;
    let bo = f32s(ckpt, &format!("{p}o_proj.bias"), hidden)?;
    let gamma = f32s(
        ckpt,
        &format!("model.layers.{l}.input_layernorm.weight"),
        hidden,
    )?;
    // Column px's dims of a head (Dc), and its rows of Wq | Wk (h < Qh: q
    // head g Qh + h; h = Qh: kv head g).
    let dl = |px: usize, c: usize| {
        let j = px % gs;
        if c < dh {
            j * dh + c
        } else {
            d / 2 + j * dh + c - dh
        }
    };
    let qrow = |px: usize, h: usize, c: usize| ((px / gs) * qh + h) * d + dl(px, c);
    let krow = |px: usize, c: usize| (px / gs) * d + dl(px, c);
    // Wq | Wk element (head h, column dim c, hidden dim x) of column px.
    let wqk = |px: usize, h: usize, c: usize, x: usize| -> u32 {
        if x >= hidden {
            0
        } else if h < qh {
            u32::from(wq[qrow(px, h, c) * hidden + x])
        } else {
            u32::from(wk[krow(px, c) * hidden + x])
        }
    };
    let wvx = |px: usize, c: usize, x: usize| -> u32 {
        if x < hidden {
            u32::from(wv[krow(px, c) * hidden + x])
        } else {
            0
        }
    };
    if gpt.qkv1 {
        // (QKV1) one matvec: a row's [q a | q v | v]; wqv, wv stay dummies
        put16(dev, "wqa", u, hc * (2 * nh * dh + dc), |py, px, out| {
            for i in 0..hc {
                let x = py * hc + i;
                for half in [0, dh] {
                    for h in 0..nh {
                        for c in 0..dh {
                            out.push(wqk(px, h, half + c, x));
                        }
                    }
                }
                for c in 0..dc {
                    out.push(wvx(px, c, x));
                }
            }
        })?;
    } else {
        for (sym, half) in [("wqa", 0), ("wqv", dh)] {
            put16(dev, sym, u, hc * nh * dh, |py, px, out| {
                for i in 0..hc {
                    for h in 0..nh {
                        for c in 0..dh {
                            out.push(wqk(px, h, half + c, py * hc + i));
                        }
                    }
                }
            })?;
        }
        put16(dev, "wv", u, hc * dc, |py, px, out| {
            for i in 0..hc {
                let x = py * hc + i;
                for c in 0..dc {
                    out.push(wvx(px, c, x));
                }
            }
        })?;
    }
    let bqk = |px: usize, h: usize, c: usize| {
        if h < qh {
            bq[qrow(px, h, c)]
        } else {
            bk[krow(px, c)]
        }
    };
    for (sym, half) in [("bqa", 0), ("bqv", dh)] {
        put32(dev, sym, u, nh * dh, |_, px, out| {
            for h in 0..nh {
                for c in 0..dh {
                    out.push(bqk(px, h, half + c).to_bits());
                }
            }
        })?;
    }
    put32(dev, "bv", u, dc, |_, px, out| {
        out.extend((0..dc).map(|c| bv[krow(px, c)].to_bits()));
    })?;
    put32(dev, "snk", u, lanes * qh, |_, px, out| {
        for _ in 0..lanes {
            out.extend((0..qh).map(|h| sinks[(px / gs) * qh + h].to_bits()));
        }
    })?;
    let col = |v: &[f32], py: usize, i: usize| {
        let x = py * hc + i;
        if x < hidden { v[x].to_bits() } else { 0 }
    };
    put32(dev, "bo", u, hc, |py, _, out| {
        out.extend((0..hc).map(|i| col(&bo, py, i)));
    })?;
    put16(dev, "wo", u, qh * dc * hc, |py, px, out| {
        for h in 0..qh {
            for c in 0..dc {
                let r = qrow(px, h, c);
                for i in 0..hc {
                    let x = py * hc + i;
                    out.push(if x < hidden {
                        u32::from(wo[x * nq + r])
                    } else {
                        0
                    });
                }
            }
        }
    })?;
    let (inv, ms) = yarn;
    for (sym, sin) in [("rot_c", false), ("rot_s", true)] {
        put32(dev, sym, u, 3 * 16 * dh, |_, px, out| {
            let j = px % gs;
            for (lv, unit) in [256u32, 16, 1].into_iter().enumerate() {
                let sc = if lv == 0 { f64::from(*ms) } else { 1.0 };
                for t in 0..16u32 {
                    for i in 0..dh {
                        let ang = f64::from(t * unit) * f64::from(inv[j * dh + i]);
                        let v = if sin { ang.sin() } else { ang.cos() } * sc;
                        out.push((v as f32).to_bits());
                    }
                }
            }
        })?;
    }
    put32(dev, "gamma1", u, hc, |py, _, out| {
        out.extend((0..hc).map(|i| col(&gamma, py, i)));
    })
}

/// Layer `l`'s MoE weights off the checkpoint (gptoss_host.moe_layer):
/// MXFP4 codes (two a byte, low nibble first), gate/up rows de-interleaved
/// (gate the even rows, up the odd), E8M0 scales, biases, the router.
pub struct MoeWeights {
    pub experts: usize,
    pub hidden: usize,
    pub inter: usize,
    pub hidden_p: usize,
    pub inter_p: usize,
    pub gu_codes: Vec<u8>,
    pub gu_scales: Vec<u8>,
    pub gu_bias: Vec<f32>,
    pub dn_codes: Vec<u8>,
    pub dn_scales: Vec<u8>,
    pub dn_bias: Vec<f32>,
    pub router: Vec<f32>,
    pub router_bias: Vec<f32>,
}

impl MoeWeights {
    pub fn read(ckpt: &Ckpt, l: usize) -> R<MoeWeights> {
        let p = format!("model.layers.{l}.mlp.");
        let (gs, gb) = ckpt.hf_u8(&format!("{p}experts.gate_up_proj_blocks"))?;
        let [e_n, i2, g, 16] = gs[..] else {
            return Err(format!("gate_up_proj_blocks is {gs:?}"));
        };
        let inter = i2 / 2;
        let hidden_p = g * 32;
        // codes [e][r (gate rows, then up)][k]: nibble k of input row r.
        let unpack = |b: &[u8], rows: usize, k: usize, order: &dyn Fn(usize) -> usize| {
            let mut c = vec![0u8; e_n * rows * k];
            for e in 0..e_n {
                for r in 0..rows {
                    let src = &b[(e * rows + order(r)) * k / 2..][..k / 2];
                    let dst = &mut c[(e * rows + r) * k..][..k];
                    for (i, &v) in src.iter().enumerate() {
                        dst[2 * i] = v & 0x0f;
                        dst[2 * i + 1] = v >> 4;
                    }
                }
            }
            c
        };
        // Gate/up row r of [gate | up]: the checkpoint's row 2r (gate) or
        // 2(r - I) + 1 (up).
        let deint = move |r: usize| {
            if r < inter {
                2 * r
            } else {
                2 * (r - inter) + 1
            }
        };
        let gu_codes = unpack(&gb, i2, hidden_p, &deint);
        let regroup = |v: &[u8], k: usize| -> Vec<u8> {
            let mut out = vec![0u8; v.len()];
            for e in 0..e_n {
                for r in 0..i2 {
                    out[(e * i2 + r) * k..][..k]
                        .copy_from_slice(&v[(e * i2 + deint(r)) * k..][..k]);
                }
            }
            out
        };
        let (ss, sc) = ckpt.hf_u8(&format!("{p}experts.gate_up_proj_scales"))?;
        if ss != [e_n, i2, g] {
            return Err(format!("gate_up_proj_scales is {ss:?}"));
        }
        let gu_scales = regroup(&sc, g);
        let gbias = f32s(ckpt, &format!("{p}experts.gate_up_proj_bias"), e_n * i2)?;
        let mut gu_bias = vec![0f32; e_n * i2];
        for e in 0..e_n {
            for r in 0..i2 {
                gu_bias[e * i2 + r] = gbias[e * i2 + deint(r)];
            }
        }
        let (ds, db) = ckpt.hf_u8(&format!("{p}experts.down_proj_blocks"))?;
        let [_, hidden, gi, 16] = ds[..] else {
            return Err(format!("down_proj_blocks is {ds:?}"));
        };
        let inter_p = gi * 32;
        let dn_codes = unpack(&db, hidden, inter_p, &|r| r);
        let (dss, dn_scales) = ckpt.hf_u8(&format!("{p}experts.down_proj_scales"))?;
        if dss != [e_n, hidden, gi] {
            return Err(format!("down_proj_scales is {dss:?}"));
        }
        Ok(MoeWeights {
            experts: e_n,
            hidden,
            inter,
            hidden_p,
            inter_p,
            gu_codes,
            gu_scales,
            gu_bias,
            dn_codes,
            dn_scales,
            dn_bias: f32s(ckpt, &format!("{p}experts.down_proj_bias"), e_n * hidden)?,
            router: f32s(ckpt, &format!("{p}router.weight"), e_n * hidden)?,
            router_bias: f32s(ckpt, &format!("{p}router.bias"), e_n)?,
        })
    }

    pub fn layer(&self) -> MoeLayer<'_> {
        MoeLayer {
            experts: self.experts,
            hidden: self.hidden,
            inter: self.inter,
            gu_codes: &self.gu_codes,
            hidden_p: self.hidden_p,
            gu_scales: &self.gu_scales,
            gu_bias: &self.gu_bias,
            dn_codes: &self.dn_codes,
            inter_p: self.inter_p,
            dn_scales: &self.dn_scales,
            dn_bias: &self.dn_bias,
            router: &self.router,
            router_bias: &self.router_bias,
        }
    }
}

/// Layer `u.layers.start`'s MoE unit (`_moe`): moe_place's tiles, the
/// router bf16, the post-attention norm by row.
pub fn place_moe(gpt: &Gpt, hidden: usize, u: &Unit, ckpt: &Ckpt, dev: &mut dyn Device) -> R<()> {
    put_moe(hidden, u, ckpt, &moe_tiles(gpt, hidden, u, ckpt)?, dev)
}

/// Layer `u.layers.start`'s MoE tiles off the checkpoint.
fn moe_tiles(gpt: &Gpt, hidden: usize, u: &Unit, ckpt: &Ckpt) -> R<MoeTiles> {
    let l = u.layers.start;
    let w = MoeWeights::read(ckpt, l)?;
    if w.experts != gpt.ne || w.hidden != hidden {
        return Err(format!(
            "layer {l}: {} experts of {}; the program's {} of {hidden}",
            w.experts, w.hidden, gpt.ne
        ));
    }
    let g = MoeGrid {
        px: u.k.px,
        py: u.k.py,
        hc: u.k.hc,
        ic: gpt.ic,
    };
    pack_moe(&w.layer(), g).map_err(|e| format!("layer {l}: {e}"))
}

fn put_moe(hidden: usize, u: &Unit, ckpt: &Ckpt, t: &MoeTiles, dev: &mut dyn Device) -> R<()> {
    let l = u.layers.start;
    let at = rect(u.x0, u.y0, u.k.px, u.k.py);
    let pes = u.k.px * u.k.py;
    let n = |v: usize| v / pes;
    dev.put("mx", at, n(t.mx.len()), &t.mx)?;
    let hn = n(t.rw.len());
    let rw: Vec<u32> =
        t.rw.chunks_exact(hn)
            .flat_map(|pe| std::iter::once(0).chain(pe.iter().map(|&x| bf16(x))))
            .collect();
    dev.put16("rw", at, hn + 1, &rw)?;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    dev.put("rb", at, n(t.rb.len()), &bits(&t.rb))?;
    dev.put("gub", at, n(t.gub.len()), &bits(&t.gub))?;
    dev.put("dnb", at, n(t.dnb.len()), &bits(&t.dnb))?;
    dev.put("mcfg", at, 2, &t.mcfg)?;
    let gamma = f32s(
        ckpt,
        &format!("model.layers.{l}.post_attention_layernorm.weight"),
        hidden,
    )?;
    let hc = u.k.hc;
    put32(dev, "gamma2", u, hc, |py, _, out| {
        out.extend((0..hc).map(|i| {
            let x = py * hc + i;
            if x < hidden { gamma[x].to_bits() } else { 0 }
        }));
    })
}

/// Layers whose MoE tiles are built at once (a layer's experts and tiles
/// take about 1.5 GB of host memory).
const MOE_THREADS: usize = 3;

/// Every unit's weights, `MOE_THREADS` layers' MoE tiles built at once.
pub fn place_units(
    gpt: &Gpt,
    hidden: usize,
    lanes: usize,
    ckpt: &Ckpt,
    dev: &mut dyn Device,
) -> R<()> {
    let yarn = yarn_inv_freq(ckpt.config())?;
    let dh = gpt.dc / 2;
    if yarn.0.len() != gpt.gs * dh {
        return Err(format!(
            "the checkpoint's head_dim is {}; the program's {}",
            2 * yarn.0.len(),
            gpt.gs * gpt.dc
        ));
    }
    let units = gpt.slots.units();
    for layers in units.chunks(2 * MOE_THREADS) {
        let tiles: Vec<R<MoeTiles>> = std::thread::scope(|s| {
            let jobs: Vec<_> = layers
                .iter()
                .filter(|u| u.kind == Kind::Moe)
                .map(|u| s.spawn(move || moe_tiles(gpt, hidden, u, ckpt)))
                .collect();
            jobs.into_iter()
                .map(|j| {
                    j.join()
                        .unwrap_or_else(|_| Err("a MoE tile thread panicked".into()))
                })
                .collect()
        });
        let mut tiles = tiles.into_iter();
        for u in layers {
            if u.kind == Kind::Attn {
                place_attn(gpt, hidden, u, ckpt, &yarn, lanes, dev)?;
            } else {
                let t = tiles.next().ok_or("a MoE unit without its tiles")??;
                put_moe(hidden, u, ckpt, &t, dev)?;
                tracing::info!(
                    layer = u.layers.start,
                    "spatial wafer: serve_gptoss layer placed"
                );
            }
        }
    }
    Ok(())
}

/// The route tables before `init`: HROUTE's collective words a unit (the
/// program's encoding read first, `rbits` at layer 0's attention unit),
/// then every unit's handoff table.
pub fn put_routes(gpt: &Gpt, lanes: usize, dev: &mut dyn Device) -> R<()> {
    let units = gpt.slots.units();
    if gpt.hroute {
        let u0 = &units[0];
        let rb = dev
            .get("rbits", rect(u0.x0, u0.y0, 1, 1), 10)
            .map_err(|e| format!("reading rbits: {e}"))?;
        for u in &units {
            let gy = if u.kind == Kind::Attn {
                gpt.agy
            } else {
                gpt.mgy
            };
            if u.kind == Kind::Attn && gpt.arf != 0 {
                // (ARF) streamed halves: rows in groups of 8, ARF 3 columns in groups of 16
                let argx = if gpt.arf & 2 != 0 { 16 } else { 0 };
                let words = route_words_arf(&rb, u.k.px, u.k.py, gpt.gs, gy, 8, argx, false);
                let n = if argx > 0 { 18 } else { 12 };
                dev.put("rcfg", rect(u.x0, u.y0, u.k.px, u.k.py), n, &words)?;
                continue;
            }
            let words = route_words(&rb, u.k.px, u.k.py, gpt.gs, gy);
            dev.put("rcfg", rect(u.x0, u.y0, u.k.px, u.k.py), 11, &words)?;
        }
    }
    let t = tables(&gpt.slots, lanes, gpt.rtr, gpt.rti)?;
    for u in &units {
        dev.put16(
            "rtab",
            rect(u.x0, u.y0, u.k.px, u.k.py),
            t.rtn(),
            &t.table(u)?,
        )?;
    }
    Ok(())
}

/// A launch's tables (`tab`, a row broadcast over each run of units side
/// by side): the attention units' `attn` (`AY` rows of the pool's
/// [`Shape::row_words`]), the MoE units' its first row's positions and
/// slots with no pages (E 1).
pub fn put_tables(gpt: &Gpt, shape: Shape, dev: &mut dyn Device, attn: &[u32]) -> R<()> {
    let s = &gpt.slots;
    let (b_n, w) = (shape.lanes, shape.row_words());
    if attn.len() != s.ay * w {
        return Err(format!(
            "an attention table of {} words, not {} x {w}",
            attn.len(),
            s.ay
        ));
    }
    let mw = 2 * b_n + 3 * b_n;
    let mut row = vec![NONE; mw];
    row[..2 * b_n].copy_from_slice(&attn[..2 * b_n]);
    let moe: Vec<u32> = (0..s.my).flat_map(|_| row.iter().copied()).collect();
    for (r, kind) in runs(&s.units()) {
        let (tab, n) = if kind == Kind::Attn {
            (attn, w)
        } else {
            (&moe[..], mw)
        };
        dev.memcpy_h2d_rowbcast("tab", tab, r, n, DataType::Bits32)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The units as rectangles of side-by-side units of one kind (no PE
/// outside a unit).
fn runs(units: &[Unit]) -> Vec<(crate::sdk::Rect, Kind)> {
    let mut by: Vec<&Unit> = units.iter().collect();
    by.sort_by_key(|u| (u.y0, u.x0));
    let mut out: Vec<(crate::sdk::Rect, Kind)> = Vec::new();
    for u in by {
        if let Some((r, k)) = out.last_mut()
            && *k == u.kind
            && r.y as usize == u.y0
            && r.h as usize == u.k.py
            && (r.x + r.w) as usize == u.x0
        {
            r.w += u.k.px as i32;
            continue;
        }
        out.push((rect(u.x0, u.y0, u.k.px, u.k.py), u.kind));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv(words: impl Iterator<Item = u32>) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for w in words {
            for b in w.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    fn hash(s: &Slots) -> (usize, u64) {
        let t = tables(s, 1, 6, 6).expect("tables");
        let mut all = Vec::new();
        for u in s.units() {
            all.extend(t.table(&u).expect("a unit's table"));
        }
        (all.len(), fnv(all.into_iter()))
    }

    /// `GPT_CMP=<dir>`: the routes and placement of `GPT_CKPT` (a Hugging
    /// Face directory, files whole or in pieces) for `<dir>/small.params`,
    /// dumped to `<dir>/rs.bin` (every PE's words of every symbol), to set
    /// against a reference placement's (`rbits` reads 0x111 (i + 1)). `GPT_FORGET`: a full-size
    /// run, every call's sha256 to `<dir>/rs_calls.txt` instead.
    #[test]
    #[ignore]
    fn dump_placement() {
        use crate::spatial::codegen::Design;
        use crate::spatial::component::Weights;
        use crate::spatial::device::Recorder;
        use crate::spatial::program::{ProgramSpec, parse_params};
        let dir = std::path::PathBuf::from(std::env::var("GPT_CMP").expect("GPT_CMP"));
        let params = parse_params(
            std::fs::read_to_string(dir.join("small.params"))
                .unwrap()
                .trim(),
        );
        let spec =
            ProgramSpec::from_params(Design::ServeGptoss, &params).expect("a serve_gptoss program");
        let ckpt = Ckpt::open(std::path::Path::new(
            &std::env::var("GPT_CKPT").expect("GPT_CKPT"),
        ))
        .expect("the checkpoint");
        let rec = if std::env::var("GPT_FORGET").is_ok() {
            Recorder::default().forgetful()
        } else {
            Recorder::default()
        };
        let gpt = spec.gptoss().unwrap();
        let u0 = &gpt.slots.units()[0];
        let rb: Vec<u32> = (1..=10).map(|i| 0x111 * i).collect();
        rec.preset("rbits", u0.x0 as i32, u0.y0 as i32, &rb);
        let mut dev = rec.clone();
        let w = Weights {
            ckpt: &ckpt,
            rope: &crate::spatial::wafer::Rotary::theta(0.0),
        };
        for c in spec.components() {
            c.routes(&spec, &mut dev).expect("routes");
        }
        let t = std::time::Instant::now();
        for c in spec.components() {
            c.place(&spec, &w, &mut dev).expect("placement");
        }
        eprintln!("placed in {:?}", t.elapsed());
        if std::env::var("GPT_FORGET").is_ok() {
            // Full size: each call's digest (calls.txt) instead of words.
            let calls: Vec<String> = rec.take();
            std::fs::write(dir.join("rs_calls.txt"), calls.join("\n") + "\n").unwrap();
            return;
        }
        let mut out = Vec::new();
        for ((sym, x, y), words) in rec.dump() {
            if sym == "rbits" {
                continue;
            }
            out.extend((sym.len() as u32).to_le_bytes());
            out.extend(sym.as_bytes());
            out.extend(x.to_le_bytes());
            out.extend(y.to_le_bytes());
            out.extend((words.len() as u32).to_le_bytes());
            for w in words {
                out.extend(w.to_le_bytes());
            }
        }
        std::fs::write(dir.join("rs.bin"), out).unwrap();
    }

    /// A launch's tables: every attention PE the pool table, every MoE PE
    /// the lanes' positions (E 1, no pages), as gptoss_host.put_tables;
    /// no PE outside a unit written (the spare slot, the head).
    #[test]
    fn a_launch_writes_each_unit_its_table() {
        use crate::spatial::device::Recorder;
        use crate::spatial::paged::{Lane, table};
        use crate::spatial::program::{ProgramSpec, parse_params};
        let params = parse_params(MINI);
        let spec = ProgramSpec::from_params(crate::spatial::codegen::Design::ServeGptoss, &params)
            .unwrap();
        let gpt = spec.gptoss().unwrap();
        let shape = spec.paged().unwrap();
        assert_eq!((shape.rows, shape.entries, shape.pages()), (5, 2, 10));
        let lane = Lane {
            pos: 9,
            slot: 9,
            pages: vec![7, 2, 4],
            visible: None,
        };
        let attn = table(&shape, std::slice::from_ref(&lane), 3).unwrap();
        let rec = Recorder::default();
        let mut dev = rec.clone();
        put_tables(&gpt, shape, &mut dev, &attn).unwrap();
        let mut moe = vec![NONE; 5];
        moe[..2].copy_from_slice(&[9, 9]);
        let units = gpt.slots.units();
        let mut pes = 0;
        for u in &units {
            for py in 0..u.k.py {
                let want = if u.kind == Kind::Attn {
                    attn[py * shape.row_words()..(py + 1) * shape.row_words()].to_vec()
                } else {
                    moe.clone()
                };
                for px in 0..u.k.px {
                    let at = ((u.x0 + px) as i32, (u.y0 + py) as i32);
                    assert_eq!(rec.memory("tab", at.0, at.1), Some(want.clone()), "{at:?}");
                    pes += 1;
                }
            }
        }
        assert_eq!(rec.dump().len(), pes);
        // Row 2 holds logical page 0 (pool page 7: block 1), its four
        // positions visible.
        let r2 = &attn[2 * shape.row_words()..3 * shape.row_words()];
        assert_eq!((r2[0], r2[1], r2[2], r2[4], r2[6]), (9, 9, 1, 0, 0xf));
        // Every slot used: each slot row's attention and MoE units as one
        // run each; at full size the spare slot splits the last row.
        assert_eq!(runs(&units).len(), 2 * 3);
        let full = Slots {
            px: 128,
            ay: 42,
            ahc: 69,
            my: 192,
            mhc: 15,
            gw: 5,
            gh: 5,
            nl: 24,
        };
        assert_eq!(runs(&full.units()).len(), 2 * 5);
    }

    /// `GPT_CKPT=<ck_mini6>`: the driver's whole host side on the mini
    /// program on a recording device: load (routes, init, placement,
    /// warm-up), a prompt launch, a decode launch with the logits read, a
    /// page copy and the host tier's page out and in; every symbol it
    /// touches is one the components declare.
    #[test]
    #[ignore]
    fn the_driver_runs_the_mini_program() {
        use crate::spatial::device::Recorder;
        use crate::spatial::program::{ProgramSpec, parse_params};
        use crate::spatial::serve2d::BatchLane;
        use crate::spatial::wafer::{WaferModel, Weights};
        let spec = ProgramSpec::from_params(
            crate::spatial::codegen::Design::ServeGptoss,
            &parse_params(MINI),
        )
        .unwrap();
        let ckpt = Ckpt::open(std::path::Path::new(
            &std::env::var("GPT_CKPT").expect("GPT_CKPT"),
        ))
        .unwrap();
        let rec = Recorder::default();
        let u0 = &spec.gptoss().unwrap().slots.units()[0];
        rec.preset("rbits", u0.x0 as i32, u0.y0 as i32, &[1; 10]);
        let declared: std::collections::BTreeSet<&str> =
            spec.abi().iter().map(|(_, s)| s.name).collect();
        let mut m = WaferModel::on_device(
            Box::new(rec.clone()),
            spec.clone(),
            Weights::Place(&ckpt),
            None,
            0,
        )
        .expect("loads");
        let lane = |pos: u32, forced: Vec<u32>| BatchLane {
            pos,
            slot: pos,
            pages: vec![3, 8, 1],
            visible: None,
            forced,
            state: None,
        };
        m.fire_batch(&[lane(0, vec![5, 6, 7])], 3, false).unwrap();
        let (_, logits) = m.fire_batch(&[lane(3, vec![9])], 1, true).unwrap();
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), 1000);
        m.copy_pages(&[(3, 4)]).unwrap();
        let out = m.page_out(&[4]).unwrap();
        m.page_in(&[(5, out[0].clone())]).unwrap();
        let calls = rec.take();
        for c in &calls {
            if let Some(sym) = c.split_whitespace().nth(1) {
                assert!(declared.contains(sym), "{c}: `{sym}` undeclared");
            }
        }
        if std::env::var("GPT_PRINT").is_ok() {
            for c in calls.iter().filter(|c| !c.contains(" wqa ")) {
                eprintln!("{c}");
            }
        }
        assert!(
            calls
                .iter()
                .any(|c| c.starts_with("launch compute") || c.starts_with("launch_nb compute")),
            "{calls:?}"
        );
    }

    /// The simulator's mini program.
    const MINI: &str = "PX:8,AY:5,AHc:13,MY:18,MHc:4,HR:64,Dc:4,Gs:4,Qh:2,Ic:5,NE:8,TK:4,\
AGY:5,MGY:6,GW:3,GH:3,NL:9,SLW:341,WIN:4,RTR:6,RTI:6,PS:4,NB:2,E:2,B:1,MC:1,NP:3,PROF:0,TIME:0,\
HROUTE:1,HH:40,Vr:25,Gx:2,Gy:4,vocab:1000,PASSES:3,MODE:0,OVL:0,VMC:1,CAP:16,iters:1";

    /// The tables equal the bare host's (gptoss_host.Tables, FNV-1a 64 of
    /// every unit's rtab words in unit order), at the simulator's mini shape
    /// and at full size (the program CS-3 ran in 0447).
    #[test]
    fn tables_match_the_bare_host() {
        let mini = Slots {
            px: 8,
            ay: 5,
            ahc: 13,
            my: 18,
            mhc: 4,
            gw: 3,
            gh: 3,
            nl: 9,
        };
        assert_eq!(hash(&mini), (96048, 0x9b14c126a5c119fd));
        let full = Slots {
            px: 128,
            ay: 42,
            ahc: 69,
            my: 192,
            mhc: 15,
            gw: 5,
            gh: 5,
            nl: 24,
        };
        assert_eq!(hash(&full), (41693184, 0xdf3663bd368add1b));
    }
}
