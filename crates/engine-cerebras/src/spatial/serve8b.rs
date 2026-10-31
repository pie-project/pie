//! The serve8b wafer program (`spatial/csl/serve8b`): Qwen3-8B on one wafer.
//! Its host side as far as serving goes (`HEADMODE` 2): the decoder layers
//! as qwen2d_paged8b grids (bf16 weights and KV, B lanes over a paged pool),
//! head2d_v10 east of them (`lm_head.weight`: an untied head), the embedding
//! strip east of the head (`embed_tokens`).
//!
//! The shape is a [`ProgramSpec`] with `Layers::Strips`: grids of `PX` x `PY` PEs,
//! `GW` across and `GH` down, serpentine as serve2d's, except that the
//! strips past the first `GS` skip their westmost slot, so the program holds
//! `GS` x `GW` plus (`GH` - `GS`) x (`GW` - 1) layers. Requests, logits and
//! `KVCP` 0 page copies are the components' ([`super::serve2d::fire`],
//! [`super::program::Kv::copy_pages`], the tables written over
//! [`ProgramSpec::grid_areas`]). What is serve8b's own is here: the grids'
//! handoff routes ([`route_table`], written before `init`), the weights'
//! placement (`place_layers`, and the head and the embedding strip shared
//! with serve8b_region), and the bf16 pool's host pages.

use super::ckpt::Ckpt;
use super::device::Device;
use super::program::ProgramSpec;
use super::wafer::{inv_freq, rect};

type R<T> = std::result::Result<T, String>;

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

/// Grid `l`'s per-PE handoff table, `PY` x `PX` PEs of [`RTN`] words:
/// layer l takes x from layer l - 1's
/// grid (the embedding strip for layer 0) and hands it to l + 1's (the head
/// for the last). A neighbour in the same strip: the edge column's row py,
/// multicast along row py (R0/R1 by parity). The grid below (a strip
/// change): the PE (py mod PX, py) down its column on CV (rows < PX) or
/// CV2 (rows past PX).
#[allow(clippy::too_many_lines)]
pub fn route_table(lay: &ProgramSpec, l: usize) -> R<Vec<u32>> {
    let (pxn, pyn) = (lay.px(), lay.py());
    let (x0, y0) = lay.origin(l);
    let prev = (l > 0).then(|| lay.origin(l - 1));
    let nxt = (l + 1 < lay.layers()).then(|| lay.origin(l + 1));
    let rc = |k: usize| if k.is_multiple_of(2) { C_R0 } else { C_R1 };
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
                    rts.push(if p.0 < x0 {
                        [c_in, D_W, D_RAMP, te]
                    } else {
                        [c_in, D_E, D_RAMP, tw]
                    });
                }
                // The grid above.
                Some(p) => {
                    if p != (x0, y0.wrapping_sub(pyn)) {
                        return Err(format!(
                            "layer {l}: the previous grid {p:?} is not above {:?}",
                            (x0, y0)
                        ));
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
                        rts.push(if px < dg {
                            [c_in, D_E, D_RAMP, tw]
                        } else {
                            [c_in, D_W, D_RAMP, te]
                        });
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
                        return Err(format!(
                            "layer {l}: the next grid {n:?} is not below {:?}",
                            (x0, y0)
                        ));
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
                return Err(format!(
                    "layer {l} PE ({px}, {py}): routes {rts:?} do not fit its table"
                ));
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

/// Every grid's handoff table: before `init`, which applies them.
pub fn put_routes(lay: &ProgramSpec, dev: &mut dyn Device) -> R<()> {
    for l in 0..lay.layers() {
        let (x0, y0) = lay.origin(l);
        dev.put16(
            "rtab",
            rect(x0, y0, lay.px(), lay.py()),
            RTN,
            &route_table(lay, l)?,
        )?;
    }
    Ok(())
}

/// A 2-D tensor's bf16 bits, its shape checked.
pub(crate) fn matrix(ckpt: &Ckpt, name: &str, rows: usize, cols: usize) -> R<Vec<u16>> {
    let (shape, bits) = ckpt.bits(name)?;
    if shape != [rows, cols] {
        return Err(format!(
            "{name} is {shape:?}; the program wants [{rows}, {cols}]"
        ));
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

/// Per-PE blocks of a `k.py` x `k.px` unit, one leading pad element a PE,
/// 16-bit: `f(py, px, out)` appends a PE's elements.
fn put_pe16(
    k: Kernel,
    dev: &mut dyn Device,
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
    dev.put16(sym, rect(at.0, at.1, k.px, k.py), n + 1, &words)
}

fn put_pe32(
    k: Kernel,
    dev: &mut dyn Device,
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
    dev.put(sym, rect(at.0, at.1, k.px, k.py), n, &words)
}

/// Every weight of `ckpt` (and zeroed KV) for serve8b, in this order: the
/// layers, the head, the embedding strip. The routes
/// ([`put_routes`]) go before `init`, not here.
pub(crate) fn place_layers(
    lay: &ProgramSpec,
    ckpt: &Ckpt,
    rope_theta: f64,
    dev: &mut dyn Device,
) -> R<()> {
    let inv_freq = inv_freq(rope_theta, lay.gs() * lay.dc());
    let rope = Rope {
        units: &[256, 16, 1],
        len: 16,
        inv_freq: &inv_freq,
    };
    let k = Kernel {
        px: lay.px(),
        py: lay.py(),
        hc: lay.hc(),
        dc: lay.dc(),
        gs: lay.gs(),
        ic: lay.ic(),
    };
    for l in 0..lay.layers() {
        unit(lay, ckpt, k, &[l], lay.origin(l), &rope, true, dev)?;
    }
    Ok(())
}

/// A unit's per-PE blocks (one layer, or a serve8b_region region of
/// several): `layers`' blocks stacked
/// `[layer][...]` in each per-layer symbol, one leading pad in the 16-bit
/// ones; the RoPE tables once. Column px holds kv head g = px / Gs's dims
/// dl (a DH run from each RoPE half), its Qh query heads over the same dims,
/// intermediate dims [px Ic, (px+1) Ic); row py holds hidden dims [py Hc,
/// (py+1) Hc) (zero past HR). bf16 bits go through untouched. `kv_first`:
/// serve8b's order (kc, vc before the norms) rather than serve8b_region's.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn unit(
    lay: &ProgramSpec,
    ckpt: &Ckpt,
    k: Kernel,
    layers: &[usize],
    at: (usize, usize),
    rope: &Rope,
    kv_first: bool,
    dev: &mut dyn Device,
) -> R<()> {
    let (pxn, hc, dc, gs, ic) = (k.px, k.hc, k.dc, k.gs, k.ic);
    let (qh, hid, nl) = (lay.qh(), lay.hidden(), layers.len());
    let (d, dh, nh) = (gs * dc, dc / 2, qh + 1);
    let nkv = pxn / gs;
    let (nq, inter) = (nkv * qh, pxn * ic);
    let name = |l: usize, w: &str| format!("layers.{l}.{w}");
    let mats = |w: &str, rows: usize, cols: usize| -> R<Vec<Vec<u16>>> {
        layers
            .iter()
            .map(|&l| matrix(ckpt, &name(l, w), rows, cols))
            .collect()
    };
    let vecs = |w: &str, n: usize| -> R<Vec<Vec<f32>>> {
        layers
            .iter()
            .map(|&l| vector(ckpt, &name(l, w), n))
            .collect()
    };
    // Column px's dims of its kv head: a DH run from each RoPE half.
    let dl = |px: usize| -> Vec<usize> {
        let j = px % gs;
        (j * dh..(j + 1) * dh)
            .chain((j * dh..(j + 1) * dh).map(|i| d / 2 + i))
            .collect()
    };
    // Hidden dim (py, hc), or None in the padding past HR.
    let dim = |py: usize, kk: usize| Some(py * hc + kk).filter(|&x| x < hid);
    let wq = mats("q", nq * d, hid)?;
    let wk = mats("k", nkv * d, hid)?;
    // Layer li's head h's (Qh: k's) row of dim dd of kv head g.
    let qk = |li: usize, g: usize, h: usize, dd: usize, col: usize| -> u32 {
        if h < qh {
            u32::from(wq[li][((g * qh + h) * d + dd) * hid + col])
        } else {
            u32::from(wk[li][(g * d + dd) * hid + col])
        }
    };
    for (sym, half) in [("wqa", 0), ("wqv", dh)] {
        put_pe16(k, dev, sym, at, nl * hc * nh * dh, |py, px, out| {
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
    let wv = mats("v", nkv * d, hid)?;
    put_pe16(k, dev, "wv", at, nl * hc * dc, |py, px, out| {
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
    let wo = mats("o", hid, nq * d)?;
    put_pe16(k, dev, "wo", at, nl * qh * dc * hc, |py, px, out| {
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
    let wg = mats("gate", inter, hid)?;
    let wu = mats("up", inter, hid)?;
    put_pe16(k, dev, "wgu", at, nl * hc * 2 * ic, |py, px, out| {
        for li in 0..nl {
            for kk in 0..hc {
                for (w, i) in [&wg[li], &wu[li]]
                    .into_iter()
                    .flat_map(|w| (0..ic).map(move |i| (w, i)))
                {
                    out.push(dim(py, kk).map_or(0, |col| u32::from(w[(px * ic + i) * hid + col])));
                }
            }
        }
    })?;
    drop((wg, wu));
    let wd = mats("down", hid, inter)?;
    put_pe16(k, dev, "wd", at, nl * ic * hc, |py, px, out| {
        for w in &wd {
            for i in 0..ic {
                for kk in 0..hc {
                    out.push(dim(py, kk).map_or(0, |row| u32::from(w[row * inter + px * ic + i])));
                }
            }
        }
    })?;
    drop(wd);
    let qn = vecs("q_norm", d)?;
    let kn = vecs("k_norm", d)?;
    let scale = (d as f64).sqrt() as f32;
    put_pe32(k, dev, "qkn_w", at, nl * 2 * nh * dh, |_, px, out| {
        let dl = dl(px);
        for li in 0..nl {
            for half in [0, dh] {
                for h in 0..nh {
                    for i in 0..dh {
                        let dd = dl[half + i];
                        out.push(
                            if h < qh {
                                qn[li][dd] / scale
                            } else {
                                kn[li][dd]
                            }
                            .to_bits(),
                        );
                    }
                }
            }
        }
    })?;
    for (sym, sin) in [("rot_c", false), ("rot_s", true)] {
        put_pe32(
            k,
            dev,
            sym,
            at,
            rope.units.len() * rope.len as usize * dh,
            |_, px, out| {
                let j = px % gs;
                for &u in rope.units {
                    for t in 0..rope.len {
                        for i in 0..dh {
                            let ang = f64::from(t * u) * f64::from(rope.inv_freq[j * dh + i]);
                            out.push((if sin { ang.sin() } else { ang.cos() } as f32).to_bits());
                        }
                    }
                }
            },
        )?;
    }
    let shape = lay.paged().ok_or("serve8b without its paged shape")?;
    let kv = nl * shape.blocks * dc * shape.page_size;
    let put_kv = |dev: &mut dyn Device| -> R<()> {
        for sym in ["kc", "vc"] {
            put_pe16(k, dev, sym, at, kv, |_, _, out| {
                out.extend(std::iter::repeat_n(0, kv))
            })?;
        }
        Ok(())
    };
    if kv_first {
        put_kv(dev)?;
    }
    for (sym, w) in [("gamma1", "attn_norm"), ("gamma2", "mlp_norm")] {
        let g = vecs(w, hid)?;
        put_pe32(k, dev, sym, at, nl * hc, |py, _, out| {
            for g in &g {
                out.extend((0..hc).map(|kk| dim(py, kk).map_or(0, |x| g[x].to_bits())));
            }
        })?;
    }
    if !kv_first {
        put_kv(dev)?;
    }
    Ok(())
}

/// Writes a table of `HH` region rows, `w` PEs of `n` words each, built
/// [`BLOCK`] rows at a time by `tile(r0, r1)`, to the region rows at
/// columns `x0..` (skipping the feeder row F, as `_region_rows`).
fn region_rows(
    lay: &ProgramSpec,
    dev: &mut dyn Device,
    sym: &str,
    x0: usize,
    w: usize,
    n: usize,
    tile: impl Fn(usize, usize) -> Vec<u32>,
) -> R<()> {
    let (f, hh) = (lay.feeder(), lay.hh());
    for r0 in (0..hh).step_by(BLOCK) {
        let r1 = hh.min(r0 + BLOCK);
        let t = tile(r0, r1);
        for (a, b) in [(r0, r1.min(f)), (r0.max(f), r1)] {
            if a < b {
                let y = if a < f { a } else { a + 1 };
                dev.put(
                    sym,
                    rect(x0, y, w, b - a),
                    n,
                    &t[(a - r0) * w * n..(b - r0) * w * n],
                )?;
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
pub(crate) fn head(lay: &ProgramSpec, ckpt: &Ckpt, dev: &mut dyn Device) -> R<()> {
    let (vr, c_n, hid, vmc) = (lay.vr(), lay.chunk_cols(), lay.hidden(), lay.vmc());
    let hh2 = 2 * lay.hc();
    let (shape, lm) = ckpt.bits("head")?;
    let v = shape[0];
    if shape.len() != 2 || shape[1] != hid || v > lay.vocab() || lay.vocab() > lay.hh() * vr {
        return Err(format!(
            "head is {shape:?}; the head holds {} rows of {hid} ({} words)",
            lay.vocab(),
            lay.hh() * vr
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
        if vocab_row < v && d < hid {
            lm[vocab_row * hid + d]
        } else {
            0
        }
    };
    region_rows(lay, dev, "wt", lay.head_x(), cols.len(), n, |r0, r1| {
        let mut t = Vec::with_capacity((r1 - r0) * cols.len() * n);
        for row in r0..r1 {
            for c in &cols {
                match c {
                    Some(c) => {
                        t.extend((0..n).map(|w| pack(el(row, *c, 2 * w), el(row, *c, 2 * w + 1))))
                    }
                    None => t.extend(std::iter::repeat_n(0, n)),
                }
            }
        }
        t
    })?;
    drop(lm);
    let g = vector(ckpt, "final_norm", hid)?;
    let gam: Vec<u32> = cols
        .iter()
        .flat_map(|c| (0..hh2).map(move |k| c.map(|c| c * hh2 + k)))
        .map(|d| d.filter(|&d| d < hid).map_or(0, |d| g[d].to_bits()))
        .collect();
    dev.put(
        "gam",
        rect(lay.head_x(), lay.feeder(), cols.len(), 1),
        hh2,
        &gam,
    )
}

/// The strip's tiles of `model.embed_tokens` (`_embed`): region PE (strip
/// column EC-1-c, ry) holds dims [c He, (c+1) He) of vocab rows ry Vr ..,
/// row-major bf16, two to a word.
pub(crate) fn embed(lay: &ProgramSpec, ckpt: &Ckpt, dev: &mut dyn Device) -> R<()> {
    let (ve, hid) = (lay.vr(), lay.hidden());
    let (ec, he) = (lay.py() / 4, 4 * lay.hc());
    let (shape, em) = ckpt.bits("embed")?;
    let v = shape[0];
    if shape.len() != 2 || shape[1] != hid || v > lay.hh() * ve {
        return Err(format!(
            "embed is {shape:?}; the strip holds {} rows of {hid}",
            lay.hh() * ve
        ));
    }
    let n = ve * he / 2;
    let el = |row: usize, c: usize, j: usize| -> u16 {
        let (vv, k) = (j / he, j % he);
        let (vocab_row, d) = (row * ve + vv, c * he + k);
        if vocab_row < v && d < hid {
            em[vocab_row * hid + d]
        } else {
            0
        }
    };
    let xe = lay.head_x() + lay.chunk_cols() + lay.vmc();
    region_rows(lay, dev, "etab", xe, ec, n, |r0, r1| {
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
pub fn copy_pages(lay: &ProgramSpec, rt: &mut dyn Device, pairs: &[(u32, u32)]) -> R<()> {
    let pages = lay.paged().map_or(0, |s| s.pages()) as u32;
    let py = lay.py() as u32;
    let at = |g: u32| (g % py) | ((g / py) << 16);
    for &(src, dst) in pairs {
        if src >= pages || dst >= pages {
            return Err(format!(
                "page copy {src} -> {dst} outside a pool of {pages}"
            ));
        }
        for phase in [0, 1] {
            rt.launch("copy_kv", &[at(src), at(dst), phase])
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// One unit's share of the pool: its corner, `px` x `py` PEs, `nl` layers,
/// `n` cells a page on a PE (PS Dc), `kcn` a layer's cells (NB n).
struct PoolUnit {
    x0: usize,
    y0: usize,
    px: usize,
    py: usize,
    nl: usize,
    n: usize,
    kcn: usize,
}

fn pool_units(lay: &ProgramSpec) -> R<Vec<PoolUnit>> {
    let shape = lay
        .paged()
        .ok_or("host page copies on a one-sequence build")?;
    let of = |x0, y0, k: Kernel, nl| {
        let n = shape.page_size * k.dc;
        PoolUnit {
            x0,
            y0,
            px: k.px,
            py: k.py,
            nl,
            n,
            kcn: shape.blocks * n,
        }
    };
    Ok(lay
        .layers
        .units_of(lay)
        .into_iter()
        .map(|(x0, y0, k, nl)| of(x0, y0, k, nl))
        .collect())
}

/// Elements (u32 words, a bf16 each) of one pool page's host copy ([`page_out`]).
pub fn page_words(lay: &ProgramSpec) -> R<usize> {
    Ok(pool_units(lay)?.iter().map(|u| 2 * u.nl * u.px * u.n).sum())
}

/// Pool page `page`'s row and block in a unit of `py` rows, checked.
fn page_at(lay: &ProgramSpec, page: u32, py: usize) -> R<(usize, usize)> {
    let pages = lay.paged().map_or(0, |s| s.pages());
    let g = page as usize;
    if g >= pages {
        return Err(format!("page {page} outside a pool of {pages}"));
    }
    Ok((g % py, g / py))
}

/// kv_copy device -> host: pool page
/// `page`'s K and V, every layer of every unit: for each unit in layer
/// order, `kc` then `vc`, [layer][column][PS Dc] bf16 bits, one a word (as
/// the 16-bit memcpy moves them). Between launches only; the device page is
/// left as it is.
pub fn page_out(lay: &ProgramSpec, dev: &mut dyn Device, page: u32) -> R<Vec<u32>> {
    if lay.kvst() > 0 {
        return page_out_many(lay, dev, &[page]).map(|mut v| v.remove(0));
    }
    let mut out = Vec::with_capacity(page_words(lay)?);
    for u in pool_units(lay)? {
        let (row, blk) = page_at(lay, page, u.py)?;
        let len = u.nl * u.kcn + 1;
        for sym in ["kc", "vc"] {
            let a = dev.get16(sym, rect(u.x0, u.y0 + row, u.px, 1), len)?;
            for l in 0..u.nl {
                let o = 1 + l * u.kcn + blk * u.n;
                for c in a.chunks_exact(len) {
                    out.extend_from_slice(&c[o..o + u.n]);
                }
            }
        }
    }
    Ok(out)
}

/// kv_copy host -> device (`page_in`): [`page_out`]'s `data` into pool page
/// `page` (any page: a suspended page may come back elsewhere).
pub fn page_in(lay: &ProgramSpec, dev: &mut dyn Device, page: u32, data: &[u32]) -> R<()> {
    if lay.kvst() > 0 {
        return page_in_many(lay, dev, &[(page, data.to_vec())]);
    }
    let want = page_words(lay)?;
    if data.len() != want {
        return Err(format!(
            "a host page of {} elements; this build's is {want}",
            data.len()
        ));
    }
    let mut at = 0;
    for u in pool_units(lay)? {
        let (row, blk) = page_at(lay, page, u.py)?;
        let len = u.nl * u.kcn + 1;
        let row_rect = rect(u.x0, u.y0 + row, u.px, 1);
        for sym in ["kc", "vc"] {
            let mut a = dev.get16(sym, row_rect, len)?;
            for l in 0..u.nl {
                let o = 1 + l * u.kcn + blk * u.n;
                for c in a.chunks_exact_mut(len) {
                    c[o..o + u.n].copy_from_slice(&data[at..at + u.n]);
                    at += u.n;
                }
            }
            dev.write16(sym, row_rect, len, &a)?;
        }
    }
    Ok(())
}

/// Staging rounds: `pages` in rounds
/// of at most KVST a row on every unit, each with its slot (serve8b: the
/// page's index among its row's pages this round; serve8b_region: the next
/// slot free on all its units' rows), in input order.
fn stage_rounds(lay: &ProgramSpec, units: &[PoolUnit], pages: &[u32]) -> Vec<Vec<(u32, usize)>> {
    let s_n = lay.kvst();
    let mut rounds = Vec::new();
    let mut left: Vec<u32> = pages.to_vec();
    while !left.is_empty() {
        let mut used: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let (mut round, mut rest) = (Vec::new(), Vec::new());
        for &p in &left {
            // Keyed by every unit's row (serve8b's grids share theirs).
            let keys: Vec<(usize, usize)> = units
                .iter()
                .enumerate()
                .map(|(i, u)| (i, p as usize % u.py))
                .collect();
            if keys.iter().all(|k| used.get(k).copied().unwrap_or(0) < s_n) {
                let slot = keys
                    .iter()
                    .map(|k| used.get(k).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                for k in keys {
                    used.insert(k, slot + 1);
                }
                round.push((p, slot));
            } else {
                rest.push(p);
            }
        }
        rounds.push(round);
        left = rest;
    }
    rounds
}

/// The rows a round's pages hold on a unit of `py` rows: the first and how
/// many (`_stage_rect`).
fn round_rows(round: &[(u32, usize)], py: usize) -> (usize, usize) {
    let rows: Vec<usize> = round.iter().map(|&(p, _)| p as usize % py).collect();
    let (lo, hi) = (
        rows.iter().min().copied().unwrap_or(0),
        rows.iter().max().copied().unwrap_or(0),
    );
    (lo, hi - lo + 1)
}

/// The staging slots' shape on a unit: cells a slot (layers x PS Dc), a
/// PE's cells and the words a PE copies (serve8b packs two cells a 32-bit
/// word, aliasing the grid scratch; serve8b_region's slots are 16-bit, one
/// cell a word).
fn slot_shape(lay: &ProgramSpec, u: &PoolUnit) -> (usize, usize, usize, bool) {
    let per = u.nl * u.n;
    let sn = lay.kvst() * per;
    let packed = lay.design.kv_staging().0 == "bf16x2";
    (per, sn, if packed { sn.div_ceil(2) } else { sn }, packed)
}

/// [`page_out`] of several pages through the staging slots: a round at a
/// time, `page_stage(p,
/// slot)` for each of its pages, then one read a unit and symbol over the
/// rows the round's pages hold. Each page's image as [`page_out`]'s, in
/// `pages`' order. Between launches only (serve8b's slots alias the grids'
/// pass scratch: no fire may run between a stage and its read).
pub fn page_out_many(lay: &ProgramSpec, dev: &mut dyn Device, pages: &[u32]) -> R<Vec<Vec<u32>>> {
    if lay.kvst() == 0 {
        return pages.iter().map(|&p| page_out(lay, dev, p)).collect();
    }
    let units = pool_units(lay)?;
    for &p in pages {
        for u in &units {
            page_at(lay, p, u.py)?;
        }
    }
    let words = page_words(lay)?;
    let mut out: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    for round in stage_rounds(lay, &units, pages) {
        for &(p, slot) in &round {
            dev.call("page_stage", &[p, slot as u32])?;
        }
        for u in &units {
            let (r0, nr) = round_rows(&round, u.py);
            let (per, _, sw, packed) = slot_shape(lay, u);
            for sym in ["kstage", "vstage"] {
                let r = rect(u.x0, u.y0 + r0, u.px, nr);
                let a = if packed {
                    dev.get(sym, r, sw)?
                } else {
                    dev.get16(sym, r, sw)?
                };
                let cell = |pe: &[u32], c: usize| {
                    if packed {
                        (pe[c / 2] >> (16 * (c % 2))) & 0xffff
                    } else {
                        pe[c]
                    }
                };
                for &(p, slot) in &round {
                    let row = p as usize % u.py - r0;
                    let img = out.entry(p).or_insert_with(|| Vec::with_capacity(words));
                    for l in 0..u.nl {
                        for x in 0..u.px {
                            let pe = &a[(row * u.px + x) * sw..(row * u.px + x + 1) * sw];
                            img.extend((0..u.n).map(|i| cell(pe, slot * per + l * u.n + i)));
                        }
                    }
                }
            }
        }
    }
    pages
        .iter()
        .map(|p| out.remove(p).ok_or_else(|| format!("page {p} named twice")))
        .collect()
}

/// [`page_in`] of several pages through the staging slots (`pages_in`): a
/// round at a time, one write a unit and symbol over the rows the round's
/// pages hold (their slots filled, the rest zero: scratch), then
/// `page_unstage(p, slot)` for each page.
pub fn page_in_many(lay: &ProgramSpec, dev: &mut dyn Device, pages: &[(u32, Vec<u32>)]) -> R<()> {
    if lay.kvst() == 0 {
        for (p, data) in pages {
            page_in(lay, dev, *p, data)?;
        }
        return Ok(());
    }
    let want = page_words(lay)?;
    if let Some((p, d)) = pages.iter().find(|(_, d)| d.len() != want) {
        return Err(format!(
            "page {p}: a host page of {} elements; this build's is {want}",
            d.len()
        ));
    }
    let units = pool_units(lay)?;
    for (p, _) in pages {
        for u in &units {
            page_at(lay, *p, u.py)?;
        }
    }
    // Where each unit's cells start in a page image.
    let mut starts = Vec::with_capacity(units.len());
    let mut at = 0;
    for u in &units {
        starts.push(at);
        at += 2 * u.nl * u.px * u.n;
    }
    let ids: Vec<u32> = pages.iter().map(|p| p.0).collect();
    for round in stage_rounds(lay, &units, &ids) {
        for (ui, u) in units.iter().enumerate() {
            let (r0, nr) = round_rows(&round, u.py);
            let (per, _, sw, packed) = slot_shape(lay, u);
            for (si, sym) in ["kstage", "vstage"].into_iter().enumerate() {
                let mut words = vec![0u32; nr * u.px * sw];
                for &(p, slot) in &round {
                    let data = &pages.iter().find(|q| q.0 == p).expect("a round's page").1;
                    let base = starts[ui] + si * u.nl * u.px * u.n;
                    let row = p as usize % u.py - r0;
                    for l in 0..u.nl {
                        for x in 0..u.px {
                            let pe = &mut words[(row * u.px + x) * sw..(row * u.px + x + 1) * sw];
                            let src =
                                &data[base + (l * u.px + x) * u.n..base + (l * u.px + x + 1) * u.n];
                            for (i, &v) in src.iter().enumerate() {
                                let c = slot * per + l * u.n + i;
                                if packed {
                                    pe[c / 2] |= (v & 0xffff) << (16 * (c % 2));
                                } else {
                                    pe[c] = v;
                                }
                            }
                        }
                    }
                }
                let r = rect(u.x0, u.y0 + r0, u.px, nr);
                if packed {
                    dev.write(sym, r, sw, &words)?
                } else {
                    dev.write16(sym, r, sw, &words)?
                }
            }
        }
        for &(p, slot) in &round {
            dev.call("page_unstage", &[p, slot as u32])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simulator build (HEADMODE 2, B 2): 5 layers in strips of
    /// 2, 2 and 1 grids.
    pub(crate) const SIM: &str = "PX:8,PY:16,Hc:5,HR:76,Dc:8,Gs:8,Qh:4,Ic:24,PS:4,NB:2,E:2,B:2,MC:1,W16:1,NP:3,\
        KV16:1,KVCP:1,PROF:0,TIME:0,GW:2,GH:3,GS:2,HEADMODE:2,HH:32,Vr:32,Gx:4,Gy:8,vocab:1000,PASSES:3,MODE:0,\
        OVL:0,VMC:1,CAP:16,iters:1";

    pub(crate) fn sim() -> ProgramSpec {
        let kv: Vec<(&str, usize)> = SIM
            .split(',')
            .map(|p| {
                p.split_once(':')
                    .map(|(k, v)| (k, v.parse().unwrap()))
                    .unwrap()
            })
            .collect();
        ProgramSpec::of_pairs(&kv).unwrap()
    }

    #[test]
    fn the_sim_build_lays_out_in_strips() {
        let lay = sim();
        assert_eq!(lay.layers(), 5);
        let o: Vec<_> = (0..5).map(|l| lay.origin(l)).collect();
        assert_eq!(o, vec![(8, 0), (0, 0), (0, 16), (8, 16), (8, 32)]);
        assert_eq!((lay.chunk_cols(), lay.r2x(), lay.root()), (8, 6, (22, 16)));
        assert_eq!(
            lay.grid_areas(),
            vec![rect(0, 0, 16, 32), rect(8, 32, 8, 16)]
        );
        // Layer 0 takes x from the strip; layer 4 (below 3) hands to the head.
        let t0 = route_table(&lay, 0).unwrap();
        assert_eq!(&t0[..8], &[0, NONE_C, C_R1, 2, 0, D_E, D_RAMP, D_RAMP]);
        let t4 = route_table(&lay, 4).unwrap();
        let pe = |t: &[u32], px: usize, py: usize| {
            t[(py * 8 + px) * RTN..(py * 8 + px + 1) * RTN].to_vec()
        };
        // From the grid above (the relay is column 1), down CV and CV2, out on S.
        assert_eq!(
            pe(&t4, 7, 1),
            vec![
                C_R0, NONE_C, C_S[1], 4, C_R0, D_W, D_RAMP, D_RAMP, C_CV, D_N, D_S, D_S, C_CV2,
                D_N, D_S, D_S, C_S[1], D_RAMP, D_E, D_E
            ]
        );
    }

    /// A device of `kc`/`vc` rows in memory, each unwritten word its own.
    fn rows() -> crate::spatial::device::Recorder {
        crate::spatial::device::Recorder::default().with_fill(|sym, x, y, i| {
            ((x * 131 + y * 7 + i as i32 + i32::from(sym == "vc") * 5) & 0xffff) as u32
        })
    }

    /// A page out and in elsewhere reads back the same; the page it came
    /// from and the other pages on the destination rows stay as they were.
    #[test]
    fn host_pages_round_trip() {
        let lay = sim();
        let mut dev = rows();
        assert_eq!(page_words(&lay).unwrap(), 5 * 2 * 8 * 4 * 8);
        let a = page_out(&lay, &mut dev, 3).unwrap();
        let other = page_out(&lay, &mut dev, 21).unwrap();
        page_in(&lay, &mut dev, 5, &a).unwrap();
        assert_eq!(page_out(&lay, &mut dev, 5).unwrap(), a);
        assert_eq!(page_out(&lay, &mut dev, 3).unwrap(), a);
        assert_eq!(page_out(&lay, &mut dev, 21).unwrap(), other);
        assert!(page_in(&lay, &mut dev, 32, &a).is_err());
        assert!(page_in(&lay, &mut dev, 1, &a[1..]).is_err());
    }

    /// The hardware Qwen3-8B build: every grid's table is built (36 grids, strips
    /// of 5 and 4: the chain's turns and the head's hand-off fit RTN).
    #[test]
    fn qwen3_8b_routes_build() {
        let kv: Vec<(&str, usize)> = "PX:128,PY:144,Hc:29,HR:4096,Dc:8,Gs:16,Qh:4,Ic:96,PS:16,E:1,W16:1,KV16:1,\
             GW:5,GH:8,GS:4,HEADMODE:2,HH:1152,Vr:132,Gx:8,VMC:1,vocab:151936,CAP:64,B:1,NB:1,KVCP:1"
            .split(',')
            .map(|p| p.split_once(':').map(|(k, v)| (k, v.parse().unwrap())).unwrap())
            .collect();
        let lay = ProgramSpec::of_pairs(&kv).unwrap();
        assert_eq!(lay.origin(19), (512, 432));
        assert_eq!(lay.origin(20), (512, 576));
        assert_eq!(
            (lay.origin(23), lay.origin(24), lay.origin(35)),
            ((128, 576), (128, 720), (512, 1008))
        );
        for l in 0..36 {
            assert_eq!(route_table(&lay, l).unwrap().len(), 144 * 128 * RTN);
        }
    }
}
