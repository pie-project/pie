//! The serve8b wafer program (`spatial/csl/serve8b`): Qwen3-8B on one wafer.
//! The Rust form of the host side of `serve_lib8b.py`, as far
//! as serving goes (`HEADMODE` 2): the decoder layers as qwen2d_paged8b
//! grids (bf16 weights and KV, B lanes over a paged pool), head2d_v10 east
//! of them (`lm_head.weight`: an untied head), the embedding strip east of
//! the head (`embed_tokens`).
//!
//! The shape is a [`Layout2d`] with [`S8b::Strips`]: grids of `PX` x `PY` PEs,
//! `GW` across and `GH` down, serpentine as serve2d's, except that the
//! strips past the first `GS` skip their westmost slot (layers = `GS` `GW`
//! + (`GH` - `GS`) (`GW` - 1)). Requests, logits and `KVCP` 0 page copies
//! are serve2d's ([`super::serve2d::fire`], [`super::serve2d::copy_pages`],
//! the tables written over [`Layout2d::grid_areas`]). What is serve8b's own
//! is here: the grids' handoff routes ([`route_table`], written before
//! `init`) and the weights' placement ([`place`]).

use super::ckpt::Ckpt;
use super::serve2d::Layout2d;
use super::wafer::Sink;
use crate::sdk::Rect;

type R<T> = std::result::Result<T, String>;

/// How serve8b lays its layers out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S8b {
    /// serve8b: one-layer grids in strips; the first `full` hold `GW`
    /// grids, the rest `GW` - 1.
    Strips { full: usize },
    /// serve8b_region: grids and several-layer regions
    /// ([`super::serve8b_region`]).
    Units(super::serve8b_region::Units),
}

/// Words of a PE's handoff table (`grid_extra.csl`'s `rtab`):
/// `[c_in, c_relay, c_out, n, (color, rx, tx1, tx2) x n]`.
pub const RTN: usize = 4 + 4 * 4;
const NONE_C: u32 = 0xff;
const D_RAMP: u32 = 0;
const D_N: u32 = 1;
const D_S: u32 = 2;
const D_E: u32 = 3;
const D_W: u32 = 4;
const C_R0: u32 = 11;
const C_R1: u32 = 12;
const C_CV: u32 = 13;
const C_CV2: u32 = 17;
/// paths.csl's S (last layer -> head), even and odd rows.
const C_S: [u32; 2] = [14, 15];
/// paths.csl's E (embedding strip -> layer 0), row % 4.
const C_E: [u32; 4] = [0, 1, 4, 13];
/// Rows of a head or strip table built and written at once.
const BLOCK: usize = 96;

pub(crate) fn rect(x: usize, y: usize, w: usize, h: usize) -> Rect {
    Rect {
        x: x as i32,
        y: y as i32,
        w: w as i32,
        h: h as i32,
    }
}

/// serve8b's shape from its params (`GS` names it; `RPX`, serve8b_region).
pub fn layout(get: &dyn Fn(&str) -> Option<usize>) -> R<Layout2d> {
    if get("RPX").is_some() {
        return super::serve8b_region::layout(get);
    }
    let p = |k: &str| -> R<usize> { get(k).ok_or_else(|| format!("no param {k}")) };
    let mode = p("HEADMODE")?;
    if mode != 2 {
        return Err(format!("HEADMODE {mode}: the engine serves serve8b with its head (HEADMODE 2)"));
    }
    if p("W16")? != 1 || p("KV16")? != 1 {
        return Err("serve8b keeps weights and KV in bf16 (W16 1, KV16 1)".into());
    }
    let (py, full) = (p("PY")?, p("GS")?);
    let lay = Layout2d {
        px: p("PX")?,
        py,
        hc: p("Hc")?,
        hidden: p("HR")?,
        dc: p("Dc")?,
        gs: p("Gs")?,
        qh: p("Qh")?,
        ic: p("Ic")?,
        tc: 0,
        gw: p("GW")?,
        gh: p("GH")?,
        hh: p("HH")?,
        vr: p("Vr")?,
        gx: p("Gx")?,
        vmc: p("VMC")?,
        vocab: p("vocab")?,
        cap: p("CAP")?,
        paged: Some(super::paged::Shape {
            lanes: p("B")?,
            rows: py,
            page_size: p("PS")?,
            blocks: p("NB")?,
            entries: p("E")?,
        }),
        mmax: None,
        host_copy: get("KVCP").unwrap_or(1) == 0,
        vm: None,
        hlm: None,
        s8b: Some(S8b::Strips { full }),
    };
    let c = lay.chunk_cols();
    let ok = py % 4 == 0
        && lay.gs > 0
        && lay.px % lay.gs == 0
        && lay.dc % 2 == 0
        && lay.gx > 0
        && c % lay.gx == 0
        && lay.vmc <= 1
        && full >= 1
        && full <= lay.gh
        && lay.gw >= 2
        && lay.hh % 2 == 0
        && lay.vr % 2 == 0
        && py * lay.hc >= lay.hidden
        && c * 2 * lay.hc >= lay.hidden;
    if !ok {
        return Err(format!("serve8b params do not make a layout: {lay:?}"));
    }
    Ok(lay)
}

/// Grid `l`'s per-PE handoff table, `PY` x `PX` PEs of [`RTN`] words
/// (`serve_lib8b.py`'s `route_table`): layer l takes x from layer l - 1's
/// grid (the embedding strip for layer 0) and hands it to l + 1's (the head
/// for the last). A neighbour in the same strip: the edge column's row py,
/// multicast along row py (R0/R1 by parity). The grid below (a strip
/// change): the PE (py mod PX, py) down its column on CV (rows < PX) or
/// CV2 (rows past PX).
#[allow(clippy::too_many_lines)]
pub fn route_table(lay: &Layout2d, l: usize) -> R<Vec<u32>> {
    let (pxn, pyn) = (lay.px, lay.py);
    let (x0, y0) = lay.origin(l);
    let prev = (l > 0).then(|| lay.origin(l - 1));
    let nxt = (l + 1 < lay.layers()).then(|| lay.origin(l + 1));
    let rc = |k: usize| if k % 2 == 0 { C_R0 } else { C_R1 };
    let mut t = vec![0u32; pyn * pxn * RTN];
    for py in 0..pyn {
        let dg = py % pxn;
        for px in 0..pxn {
            let tw = if px > 0 { D_W } else { D_RAMP };
            let te = if px < pxn - 1 { D_E } else { D_RAMP };
            let mut rts: Vec<[u32; 4]> = Vec::new();
            let mut c_in;
            let (mut c_relay, mut c_out) = (NONE_C, NONE_C);
            match prev {
                None => {
                    c_in = C_E[py % 4];
                    rts.push([c_in, D_E, D_RAMP, tw]);
                }
                // The same strip.
                Some(p) if p.1 == y0 => {
                    c_in = rc(l);
                    rts.push(if p.0 < x0 { [c_in, D_W, D_RAMP, te] } else { [c_in, D_E, D_RAMP, tw] });
                }
                // The grid above.
                Some(p) => {
                    if p != (x0, y0.wrapping_sub(pyn)) {
                        return Err(format!("layer {l}: the previous grid {p:?} is not above {:?}", (x0, y0)));
                    }
                    c_in = rc(l);
                    if px == dg {
                        // The relay multicasts both ways (one way at a grid
                        // edge; never back to its own ramp).
                        let a = if px > 0 { tw } else { te };
                        let b = if px < pxn - 1 { te } else { tw };
                        rts.push([c_in, D_RAMP, a, b]);
                        c_relay = c_in;
                        c_in = if py < pxn { C_CV } else { C_CV2 };
                    } else {
                        rts.push(if px < dg { [c_in, D_E, D_RAMP, tw] } else { [c_in, D_W, D_RAMP, te] });
                    }
                    if py < px {
                        rts.push([C_CV, D_N, D_S, D_S]);
                    } else if py == px {
                        rts.push([C_CV, D_N, D_RAMP, D_RAMP]);
                    }
                    if px + pxn < pyn {
                        if py < px + pxn {
                            rts.push([C_CV2, D_N, D_S, D_S]);
                        } else if py == px + pxn {
                            rts.push([C_CV2, D_N, D_RAMP, D_RAMP]);
                        }
                    }
                }
            }
            match nxt {
                None => {
                    if px == pxn - 1 {
                        c_out = C_S[py % 2];
                        rts.push([c_out, D_RAMP, D_E, D_E]);
                    }
                }
                Some(n) if n.1 == y0 => {
                    if (n.0 > x0 && px == pxn - 1) || (n.0 < x0 && px == 0) {
                        c_out = rc(l + 1);
                        let d = if n.0 > x0 { D_E } else { D_W };
                        rts.push([c_out, D_RAMP, d, d]);
                    }
                }
                Some(n) => {
                    if n != (x0, y0 + pyn) {
                        return Err(format!("layer {l}: the next grid {n:?} is not below {:?}", (x0, y0)));
                    }
                    if py == px {
                        rts.push([C_CV, D_RAMP, D_S, D_S]);
                    } else if py > px {
                        rts.push([C_CV, D_N, D_S, D_S]);
                    }
                    if px + pxn < pyn {
                        if py == px + pxn {
                            rts.push([C_CV2, D_RAMP, D_S, D_S]);
                        } else if py > px + pxn {
                            rts.push([C_CV2, D_N, D_S, D_S]);
                        }
                    }
                    if px == dg {
                        c_out = if py < pxn { C_CV } else { C_CV2 };
                    }
                }
            }
            // At most four routes, and one route a color.
            let mut colors: Vec<u32> = rts.iter().map(|r| r[0]).collect();
            colors.sort_unstable();
            colors.dedup();
            if rts.len() > (RTN - 4) / 4 || colors.len() != rts.len() {
                return Err(format!("layer {l} PE ({px}, {py}): routes {rts:?} do not fit its table"));
            }
            let at = (py * pxn + px) * RTN;
            t[at..at + 4].copy_from_slice(&[c_in, c_relay, c_out, rts.len() as u32]);
            for (i, r) in rts.iter().enumerate() {
                t[at + 4 + 4 * i..at + 8 + 4 * i].copy_from_slice(r);
            }
        }
    }
    Ok(t)
}

/// Where the program's route encoding (`rbits`, 10 words) is read before
/// [`put_routes`]: serve8b_region with `HROUTE` 1.
pub fn rbits_at(lay: &Layout2d) -> Option<(usize, usize)> {
    match lay.s8b {
        Some(S8b::Units(u)) if u.hroute => Some((u.rc * u.rpx, 0)),
        _ => None,
    }
}

/// Every unit's handoff table (and with `HROUTE`, from `rbits`, its
/// collectives' routes): before `init`, which applies them.
pub fn put_routes(lay: &Layout2d, sink: &mut dyn Sink, rbits: Option<&[u32]>) -> R<()> {
    if let Some(S8b::Units(u)) = lay.s8b {
        return super::serve8b_region::put_routes(lay, &u, sink, rbits);
    }
    for l in 0..lay.layers() {
        let (x0, y0) = lay.origin(l);
        sink.h2d16("rtab", rect(x0, y0, lay.px, lay.py), RTN, &route_table(lay, l)?)?;
    }
    Ok(())
}

/// A 2-D tensor's bf16 bits, its shape checked.
pub(crate) fn matrix(ckpt: &Ckpt, name: &str, rows: usize, cols: usize) -> R<Vec<u16>> {
    let (shape, bits) = ckpt.bits(name)?;
    if shape != [rows, cols] {
        return Err(format!("{name} is {shape:?}; the program wants [{rows}, {cols}]"));
    }
    Ok(bits)
}

pub(crate) fn vector(ckpt: &Ckpt, name: &str, n: usize) -> R<Vec<f32>> {
    let t = ckpt.get(name)?;
    if t.shape != [n] {
        return Err(format!("{name} is {:?}; the program wants [{n}]", t.shape));
    }
    Ok(t.data)
}

/// A layer unit's kernel shape: `px` x `py` PEs, `hc` hidden dims a row,
/// `dc` of a kv head's dims a column (`gs` columns a kv head), `ic`
/// intermediate dims a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kernel {
    pub px: usize,
    pub py: usize,
    pub hc: usize,
    pub dc: usize,
    pub gs: usize,
    pub ic: usize,
}

/// The RoPE tables' levels: entry t of level u is position t `units[u]`
/// (3 x 16 by default, 4 x 8 with `ROT8`).
#[derive(Debug, Clone, Copy)]
pub struct Rope<'a> {
    pub units: &'a [u32],
    pub len: u32,
    pub inv_freq: &'a [f32],
}

/// RoPE's inverse frequencies (f32, as numpy rounds them).
pub(crate) fn inv_freq(rope_theta: f64, d: usize) -> Vec<f32> {
    (0..d / 2).map(|k| rope_theta.powf(-((2 * k) as f64) / d as f64) as f32).collect()
}

/// Per-PE blocks of a `k.py` x `k.px` unit, one leading pad element a PE,
/// 16-bit: `f(py, px, out)` appends a PE's elements.
fn put_pe16(
    k: Kernel,
    sink: &mut dyn Sink,
    sym: &str,
    at: (usize, usize),
    n: usize,
    f: impl Fn(usize, usize, &mut Vec<u32>),
) -> R<()> {
    let mut words = Vec::with_capacity(k.px * k.py * (n + 1));
    for py in 0..k.py {
        for px in 0..k.px {
            words.push(0);
            let start = words.len();
            f(py, px, &mut words);
            debug_assert_eq!(words.len() - start, n, "{sym}");
        }
    }
    sink.h2d16(sym, rect(at.0, at.1, k.px, k.py), n + 1, &words)
}

fn put_pe32(
    k: Kernel,
    sink: &mut dyn Sink,
    sym: &str,
    at: (usize, usize),
    n: usize,
    f: impl Fn(usize, usize, &mut Vec<u32>),
) -> R<()> {
    let mut words = Vec::with_capacity(k.px * k.py * n);
    for py in 0..k.py {
        for px in 0..k.px {
            f(py, px, &mut words);
        }
    }
    sink.h2d(sym, rect(at.0, at.1, k.px, k.py), n, &words)
}

/// Every weight of `ckpt` (and zeroed KV) for serve8b, in `serve_lib8b.py`'s
/// order: the layers, the head, the embedding strip. The routes
/// ([`put_routes`]) go before `init`, not here.
pub fn place(lay: &Layout2d, ckpt: &Ckpt, rope_theta: f64, sink: &mut dyn Sink) -> R<()> {
    if let Some(S8b::Units(u)) = lay.s8b {
        return super::serve8b_region::place(lay, &u, ckpt, rope_theta, sink);
    }
    let inv_freq = inv_freq(rope_theta, lay.gs * lay.dc);
    let rope = Rope { units: &[256, 16, 1], len: 16, inv_freq: &inv_freq };
    let k = Kernel { px: lay.px, py: lay.py, hc: lay.hc, dc: lay.dc, gs: lay.gs, ic: lay.ic };
    for l in 0..lay.layers() {
        unit(lay, ckpt, k, &[l], lay.origin(l), &rope, true, sink)?;
    }
    head(lay, ckpt, sink)?;
    embed(lay, ckpt, sink)
}

/// A unit's per-PE blocks (`serve_lib8b.py`'s `_layer`, or serve8b_region's
/// `_tile` and `_unit` for several layers): `layers`' blocks stacked
/// `[layer][...]` in each per-layer symbol, one leading pad in the 16-bit
/// ones; the RoPE tables once. Column px holds kv head g = px / Gs's dims
/// dl (a DH run from each RoPE half), its Qh query heads over the same dims,
/// intermediate dims [px Ic, (px+1) Ic); row py holds hidden dims [py Hc,
/// (py+1) Hc) (zero past HR). bf16 bits go through untouched. `kv_first`:
/// serve8b's order (kc, vc before the norms) rather than serve8b_region's.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn unit(
    lay: &Layout2d,
    ckpt: &Ckpt,
    k: Kernel,
    layers: &[usize],
    at: (usize, usize),
    rope: &Rope,
    kv_first: bool,
    sink: &mut dyn Sink,
) -> R<()> {
    let (pxn, hc, dc, gs, ic) = (k.px, k.hc, k.dc, k.gs, k.ic);
    let (qh, hid, nl) = (lay.qh, lay.hidden, layers.len());
    let (d, dh, nh) = (gs * dc, dc / 2, qh + 1);
    let nkv = pxn / gs;
    let (nq, inter) = (nkv * qh, pxn * ic);
    let name = |l: usize, w: &str| format!("model.layers.{l}.{w}");
    let mats = |w: &str, rows: usize, cols: usize| -> R<Vec<Vec<u16>>> {
        layers.iter().map(|&l| matrix(ckpt, &name(l, w), rows, cols)).collect()
    };
    let vecs = |w: &str, n: usize| -> R<Vec<Vec<f32>>> { layers.iter().map(|&l| vector(ckpt, &name(l, w), n)).collect() };
    // Column px's dims of its kv head: a DH run from each RoPE half.
    let dl = |px: usize| -> Vec<usize> {
        let j = px % gs;
        (j * dh..(j + 1) * dh).chain((j * dh..(j + 1) * dh).map(|i| d / 2 + i)).collect()
    };
    // Hidden dim (py, hc), or None in the padding past HR.
    let dim = |py: usize, kk: usize| Some(py * hc + kk).filter(|&x| x < hid);
    let wq = mats("self_attn.q_proj.weight", nq * d, hid)?;
    let wk = mats("self_attn.k_proj.weight", nkv * d, hid)?;
    // Layer li's head h's (Qh: k's) row of dim dd of kv head g.
    let qk = |li: usize, g: usize, h: usize, dd: usize, col: usize| -> u32 {
        if h < qh {
            u32::from(wq[li][((g * qh + h) * d + dd) * hid + col])
        } else {
            u32::from(wk[li][(g * d + dd) * hid + col])
        }
    };
    for (sym, half) in [("wqa", 0), ("wqv", dh)] {
        put_pe16(k, sink, sym, at, nl * hc * nh * dh, |py, px, out| {
            let (g, dl) = (px / gs, dl(px));
            for li in 0..nl {
                for kk in 0..hc {
                    for h in 0..nh {
                        for i in 0..dh {
                            out.push(dim(py, kk).map_or(0, |col| qk(li, g, h, dl[half + i], col)));
                        }
                    }
                }
            }
        })?;
    }
    drop((wq, wk));
    let wv = mats("self_attn.v_proj.weight", nkv * d, hid)?;
    put_pe16(k, sink, "wv", at, nl * hc * dc, |py, px, out| {
        let (g, dl) = (px / gs, dl(px));
        for w in &wv {
            for kk in 0..hc {
                for &dd in &dl {
                    out.push(dim(py, kk).map_or(0, |col| u32::from(w[(g * d + dd) * hid + col])));
                }
            }
        }
    })?;
    drop(wv);
    let wo = mats("self_attn.o_proj.weight", hid, nq * d)?;
    put_pe16(k, sink, "wo", at, nl * qh * dc * hc, |py, px, out| {
        let (g, dl) = (px / gs, dl(px));
        for w in &wo {
            for h in 0..qh {
                for &dd in &dl {
                    let c = (g * qh + h) * d + dd;
                    for kk in 0..hc {
                        out.push(dim(py, kk).map_or(0, |row| u32::from(w[row * nq * d + c])));
                    }
                }
            }
        }
    })?;
    drop(wo);
    let wg = mats("mlp.gate_proj.weight", inter, hid)?;
    let wu = mats("mlp.up_proj.weight", inter, hid)?;
    put_pe16(k, sink, "wgu", at, nl * hc * 2 * ic, |py, px, out| {
        for li in 0..nl {
            for kk in 0..hc {
                for (w, i) in [&wg[li], &wu[li]].into_iter().flat_map(|w| (0..ic).map(move |i| (w, i))) {
                    out.push(dim(py, kk).map_or(0, |col| u32::from(w[(px * ic + i) * hid + col])));
                }
            }
        }
    })?;
    drop((wg, wu));
    let wd = mats("mlp.down_proj.weight", hid, inter)?;
    put_pe16(k, sink, "wd", at, nl * ic * hc, |py, px, out| {
        for w in &wd {
            for i in 0..ic {
                for kk in 0..hc {
                    out.push(dim(py, kk).map_or(0, |row| u32::from(w[row * inter + px * ic + i])));
                }
            }
        }
    })?;
    drop(wd);
    let qn = vecs("self_attn.q_norm.weight", d)?;
    let kn = vecs("self_attn.k_norm.weight", d)?;
    let scale = (d as f64).sqrt() as f32;
    put_pe32(k, sink, "qkn_w", at, nl * 2 * nh * dh, |_, px, out| {
        let dl = dl(px);
        for li in 0..nl {
            for half in [0, dh] {
                for h in 0..nh {
                    for i in 0..dh {
                        let dd = dl[half + i];
                        out.push(if h < qh { qn[li][dd] / scale } else { kn[li][dd] }.to_bits());
                    }
                }
            }
        }
    })?;
    for (sym, sin) in [("rot_c", false), ("rot_s", true)] {
        put_pe32(k, sink, sym, at, rope.units.len() * rope.len as usize * dh, |_, px, out| {
            let j = px % gs;
            for &u in rope.units {
                for t in 0..rope.len {
                    for i in 0..dh {
                        let ang = f64::from(t * u) * f64::from(rope.inv_freq[j * dh + i]);
                        out.push((if sin { ang.sin() } else { ang.cos() } as f32).to_bits());
                    }
                }
            }
        })?;
    }
    let shape = lay.paged.ok_or("serve8b without its paged shape")?;
    let kv = nl * shape.blocks * dc * shape.page_size;
    let put_kv = |sink: &mut dyn Sink| -> R<()> {
        for sym in ["kc", "vc"] {
            put_pe16(k, sink, sym, at, kv, |_, _, out| out.extend(std::iter::repeat_n(0, kv)))?;
        }
        Ok(())
    };
    if kv_first {
        put_kv(sink)?;
    }
    for (sym, w) in [("gamma1", "input_layernorm.weight"), ("gamma2", "post_attention_layernorm.weight")] {
        let g = vecs(w, hid)?;
        put_pe32(k, sink, sym, at, nl * hc, |py, _, out| {
            for g in &g {
                out.extend((0..hc).map(|kk| dim(py, kk).map_or(0, |x| g[x].to_bits())));
            }
        })?;
    }
    if !kv_first {
        put_kv(sink)?;
    }
    Ok(())
}

/// Writes a table of `HH` region rows, `w` PEs of `n` words each, built
/// [`BLOCK`] rows at a time by `tile(r0, r1)`, to the region rows at
/// columns `x0..` (skipping the feeder row F, as `_region_rows`).
fn region_rows(
    lay: &Layout2d,
    sink: &mut dyn Sink,
    sym: &str,
    x0: usize,
    w: usize,
    n: usize,
    tile: impl Fn(usize, usize) -> Vec<u32>,
) -> R<()> {
    let (f, hh) = (lay.feeder(), lay.hh);
    for r0 in (0..hh).step_by(BLOCK) {
        let r1 = hh.min(r0 + BLOCK);
        let t = tile(r0, r1);
        for (a, b) in [(r0, r1.min(f)), (r0.max(f), r1)] {
            if a < b {
                let y = if a < f { a } else { a + 1 };
                sink.h2d(sym, rect(x0, y, w, b - a), n, &t[(a - r0) * w * n..(b - r0) * w * n])?;
            }
        }
    }
    Ok(())
}

/// Two bf16 to a u32 word, element 2k in the low half.
fn pack(lo: u16, hi: u16) -> u32 {
    u32::from(lo) | (u32::from(hi) << 16)
}

/// head2d_v10's bf16 tiles of `lm_head.weight` (`_head`): region PE (c, ry)
/// holds dims c Hh .. of vocab rows ry Vr .., dim-major, two to a word; a
/// zero VM column after the root column; the final norm on the feeder row.
pub(crate) fn head(lay: &Layout2d, ckpt: &Ckpt, sink: &mut dyn Sink) -> R<()> {
    let (vr, c_n, hid, vmc) = (lay.vr, lay.chunk_cols(), lay.hidden, lay.vmc);
    let hh2 = 2 * lay.hc;
    let (shape, lm) = ckpt.bits("lm_head.weight")?;
    let v = shape[0];
    if shape.len() != 2 || shape[1] != hid || v > lay.vocab || lay.vocab > lay.hh * vr {
        return Err(format!(
            "lm_head.weight is {shape:?}; the head holds {} rows of {hid} ({} words)",
            lay.vocab,
            lay.hh * vr
        ));
    }
    let r2x = lay.r2x();
    // Head columns (VM included): chunk column c, or None for the VM's.
    let cols: Vec<Option<usize>> = (0..c_n)
        .flat_map(|c| {
            let vm = if c == r2x { vmc } else { 0 };
            std::iter::once(Some(c)).chain(std::iter::repeat_n(None, vm))
        })
        .collect();
    let n = hh2 * vr / 2;
    let el = |row: usize, c: usize, j: usize| -> u16 {
        let (k, vv) = (j / vr, j % vr);
        let (vocab_row, d) = (row * vr + vv, c * hh2 + k);
        if vocab_row < v && d < hid { lm[vocab_row * hid + d] } else { 0 }
    };
    region_rows(lay, sink, "wt", lay.head_x(), cols.len(), n, |r0, r1| {
        let mut t = Vec::with_capacity((r1 - r0) * cols.len() * n);
        for row in r0..r1 {
            for c in &cols {
                match c {
                    Some(c) => t.extend((0..n).map(|w| pack(el(row, *c, 2 * w), el(row, *c, 2 * w + 1)))),
                    None => t.extend(std::iter::repeat_n(0, n)),
                }
            }
        }
        t
    })?;
    drop(lm);
    let g = vector(ckpt, "model.norm.weight", hid)?;
    let gam: Vec<u32> = cols
        .iter()
        .flat_map(|c| (0..hh2).map(move |k| c.map(|c| c * hh2 + k)))
        .map(|d| d.filter(|&d| d < hid).map_or(0, |d| g[d].to_bits()))
        .collect();
    sink.h2d("gam", rect(lay.head_x(), lay.feeder(), cols.len(), 1), hh2, &gam)
}

/// The strip's tiles of `model.embed_tokens` (`_embed`): region PE (strip
/// column EC-1-c, ry) holds dims [c He, (c+1) He) of vocab rows ry Vr ..,
/// row-major bf16, two to a word.
pub(crate) fn embed(lay: &Layout2d, ckpt: &Ckpt, sink: &mut dyn Sink) -> R<()> {
    let (ve, hid) = (lay.vr, lay.hidden);
    let (ec, he) = (lay.py / 4, 4 * lay.hc);
    let (shape, em) = ckpt.bits("model.embed_tokens.weight")?;
    let v = shape[0];
    if shape.len() != 2 || shape[1] != hid || v > lay.hh * ve {
        return Err(format!("model.embed_tokens.weight is {shape:?}; the strip holds {} rows of {hid}", lay.hh * ve));
    }
    let n = ve * he / 2;
    let el = |row: usize, c: usize, j: usize| -> u16 {
        let (vv, k) = (j / he, j % he);
        let (vocab_row, d) = (row * ve + vv, c * he + k);
        if vocab_row < v && d < hid { em[vocab_row * hid + d] } else { 0 }
    };
    let xe = lay.head_x() + lay.chunk_cols() + lay.vmc;
    region_rows(lay, sink, "etab", xe, ec, n, |r0, r1| {
        let mut t = Vec::with_capacity((r1 - r0) * ec * n);
        for row in r0..r1 {
            for s in 0..ec {
                let c = ec - 1 - s;
                t.extend((0..n).map(|w| pack(el(row, c, 2 * w), el(row, c, 2 * w + 1))));
            }
        }
        t
    })
}

/// `copy_kv`'s whole pages on a `KVCP` 1 build: `copy_kv(src, dst, phase)`
/// for phases 0 and 1 (the routes, then the move), a page named by its row
/// and block (`g % PY | (g / PY) << 16`).
pub fn copy_pages(lay: &Layout2d, rt: &mut crate::sdk::Runtime, pairs: &[(u32, u32)]) -> R<()> {
    let pages = lay.paged.map_or(0, |s| s.pages()) as u32;
    let py = lay.py as u32;
    let at = |g: u32| (g % py) | ((g / py) << 16);
    for &(src, dst) in pairs {
        if src >= pages || dst >= pages {
            return Err(format!("page copy {src} -> {dst} outside a pool of {pages}"));
        }
        for phase in [0, 1] {
            rt.launch("copy_kv", &[at(src), at(dst), phase]).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bring-up's simulator build (HEADMODE 2, B 2): 5 layers in strips of
    /// 2, 2 and 1 grids.
    pub(crate) const SIM: &str = "PX:8,PY:16,Hc:5,HR:76,Dc:8,Gs:8,Qh:4,Ic:24,PS:4,NB:2,E:2,B:2,MC:1,W16:1,NP:3,\
        KV16:1,KVCP:1,PROF:0,TIME:0,GW:2,GH:3,GS:2,HEADMODE:2,HH:32,Vr:32,Gx:4,Gy:8,vocab:1000,PASSES:3,MODE:0,\
        OVL:0,VMC:1,CAP:16,iters:1";

    pub(crate) fn sim() -> Layout2d {
        let kv: Vec<(&str, usize)> = SIM
            .split(',')
            .map(|p| p.split_once(':').map(|(k, v)| (k, v.parse().unwrap())).unwrap())
            .collect();
        Layout2d::of(|k| kv.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap().unwrap()
    }

    #[test]
    fn the_sim_build_lays_out_as_serve_lib8b() {
        let lay = sim();
        assert_eq!(lay.layers(), 5);
        let o: Vec<_> = (0..5).map(|l| lay.origin(l)).collect();
        assert_eq!(o, vec![(8, 0), (0, 0), (0, 16), (8, 16), (8, 32)]);
        assert_eq!((lay.chunk_cols(), lay.r2x(), lay.root()), (8, 6, (22, 16)));
        assert_eq!(lay.grid_areas(), vec![rect(0, 0, 16, 32), rect(8, 32, 8, 16)]);
        // Layer 0 takes x from the strip; layer 4 (below 3) hands to the head.
        let t0 = route_table(&lay, 0).unwrap();
        assert_eq!(&t0[..8], &[0, NONE_C, C_R1, 2, 0, D_E, D_RAMP, D_RAMP]);
        let t4 = route_table(&lay, 4).unwrap();
        let pe = |t: &[u32], px: usize, py: usize| t[(py * 8 + px) * RTN..(py * 8 + px + 1) * RTN].to_vec();
        // From the grid above (the relay is column 1), down CV and CV2, out on S.
        assert_eq!(
            pe(&t4, 7, 1),
            vec![C_R0, NONE_C, C_S[1], 4, C_R0, D_W, D_RAMP, D_RAMP, C_CV, D_N, D_S, D_S, C_CV2, D_N, D_S, D_S, C_S[1], D_RAMP, D_E, D_E]
        );
    }

    /// 0115's Qwen3-8B build: every grid's table is built (36 grids, strips
    /// of 5 and 4: the chain's turns and the head's hand-off fit RTN).
    #[test]
    fn qwen3_8b_routes_build() {
        let kv: Vec<(&str, usize)> = "PX:128,PY:144,Hc:29,HR:4096,Dc:8,Gs:16,Qh:4,Ic:96,PS:16,E:1,W16:1,KV16:1,\
             GW:5,GH:8,GS:4,HEADMODE:2,HH:1152,Vr:132,Gx:8,VMC:1,vocab:151936,CAP:64,B:1,NB:1,KVCP:1"
            .split(',')
            .map(|p| p.split_once(':').map(|(k, v)| (k, v.parse().unwrap())).unwrap())
            .collect();
        let lay = Layout2d::of(|k| kv.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap().unwrap();
        assert_eq!(lay.origin(19), (512, 432));
        assert_eq!(lay.origin(20), (512, 576));
        assert_eq!((lay.origin(23), lay.origin(24), lay.origin(35)), ((128, 576), (128, 720), (512, 1008)));
        for l in 0..36 {
            assert_eq!(route_table(&lay, l).unwrap().len(), 144 * 128 * RTN);
        }
    }
}
