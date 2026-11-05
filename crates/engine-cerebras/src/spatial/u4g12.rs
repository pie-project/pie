//! Gemma 4 dense in u4g64 on serve8b_region (`codegen::serve8b_region_g12`,
//! knob `G12 1`): the host side of `gemma4_unit.csl`, the layer units'
//! weights, norms, RoPE tables and route words. The head and the strip are
//! serve8b_region's W4H ones (tied: the embedding's codes), the strip scaling
//! the row by sqrt(hidden) (`ESB`).
//!
//! Layer kinds (bit l of `KINDS`): sliding, the program's `Qh` query heads a kv
//! head of `Gs` columns (head_dim `Gs Dc`), the window `WIN`; global, `QhG` a kv
//! copy of `GsG` columns: the one global kv head (Gemma-4-12B: 512 dims) is
//! placed on every copy of `GsG` columns (`PX / GsG` copies), copy r serving
//! query heads `r QhG ..`. A global layer's v is its raw k (`attention_k_eq_v`):
//! its v columns read the k projection.
//!
//! A unit PE (column px, row py): hidden rows `[py Hc, (py+1) Hc)`; column px
//! holds its kv head's (copy's) dims as RoPE pairs for q and k (a DH run from
//! each half, `px % gs`), and `Dc` consecutive dims of it for v (so a query
//! head's o rows are one run: wo's groups); its query heads' dims the same;
//! intermediate dims `[px Ic, (px+1) Ic)`. The unit's lane layout holds its most
//! query heads (QM): `qkn_w` `[layer][half][QM + 1][DH]` with a layer's own k
//! after its query heads.
//!
//! u4g64 symbols (the kernel's tables, gemma4_unit.csl's u4 consts): `wqa` the
//! fused `[q|k first halves | second halves | v]`, `wo`, `wgu` gate and up,
//! `wd`; each the headers of its distinct tiles among the unit's kinds
//! (sliding's first), then each layer's words in layer order (wgu: a layer's
//! gate words, then its up words, one header).

use super::ckpt::Ckpt;
use super::device::Device;
use super::program::ProgramSpec;
use super::serve8b::{Kernel, put_pe16};
use super::serve8b_region::{Kind, Unit, Units, route_words};
use super::u4::{Col, Q4, Tile, blob_in, run_groups};
use super::wafer::rect;

type R<T> = std::result::Result<T, String>;

/// The RoPE tables' levels (gemma4_unit.csl's `RB`): positions < 2^RB.
pub const RB: usize = 12;
/// The route words a unit PE's `rcfg` holds: the sliding kind's X, Y and
/// group broadcast (`route_words`), then the global kind's X and group broadcast.
pub const RCFG: usize = 17;

/// A G12 program's Gemma knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G12 {
    /// Bit l: layer l reads globally.
    pub kinds: u64,
    /// The global kind's query heads a kv copy and columns a copy.
    pub qhg: usize,
    pub gsg: usize,
    /// The sliding layers' window (0: every slot).
    pub win: usize,
    /// The u4 decode buffer (u16 words).
    pub w4buf: usize,
    /// f32 bits of the embedding's scale (sqrt(hidden)).
    pub esb: u32,
}

impl G12 {
    #[must_use]
    pub fn global(&self, l: usize) -> bool {
        l < 64 && self.kinds >> l & 1 == 1
    }
}

/// A layer kind's shape on a unit: query heads a kv head (copy), its
/// columns, head_dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindShape {
    pub glob: bool,
    pub qh: usize,
    pub gs: usize,
}

/// The two kinds at the program's shapes (sliding: `Qh`, `Gs`; global:
/// `QhG`, `GsG`).
#[must_use]
pub fn kinds(qh: usize, gs: usize, g: &G12) -> [KindShape; 2] {
    [
        KindShape {
            glob: false,
            qh,
            gs,
        },
        KindShape {
            glob: true,
            qh: if g.qhg > 0 { g.qhg } else { qh },
            gs: if g.gsg > 0 { g.gsg } else { gs },
        },
    ]
}

/// A kind's five tiles on a unit of `k`'s shape (gemma4_unit.csl's `W4Rv`,
/// `W4Nv`, `W4Gv`): 0 fused QKV, 1 O, 2 gate, 3 up, 4 down.
#[must_use]
pub fn tiles(k: Kernel, s: KindShape) -> [Tile; 5] {
    let (hc, dc, ic, dh) = (k.hc, k.dc, k.ic, k.dc / 2);
    let gh = run_groups(hc);
    [
        Tile {
            r: hc,
            n: 2 * (s.qh + 1) * dh + dc,
            g: gh,
        },
        Tile {
            r: s.qh * dc,
            n: hc,
            g: s.qh * run_groups(dc),
        },
        Tile {
            r: hc,
            n: ic,
            g: gh,
        },
        Tile {
            r: hc,
            n: ic,
            g: gh,
        },
        Tile {
            r: ic,
            n: hc,
            g: run_groups(ic),
        },
    ]
}

/// Symbol `m`'s (0 wqa, 1 wo, 2 wgu, 4 wd) u16 words past its pad on a unit
/// whose layers read `glob` (gemma4_unit.csl's `w4words`).
#[must_use]
pub fn symbol_words(k: Kernel, ks: &[KindShape; 2], glob: &[bool], buf: usize, m: usize) -> usize {
    let t = [tiles(k, ks[0]), tiles(k, ks[1])];
    let (has_s, has_g) = (glob.iter().any(|g| !g), glob.iter().any(|g| *g));
    let hdrs = if has_s && has_g && t[0][m] != t[1][m] {
        t[0][m].header_in(buf) + t[1][m].header_in(buf)
    } else if has_g && !has_s {
        t[1][m].header_in(buf)
    } else {
        t[0][m].header_in(buf)
    };
    let per = |g: bool| -> usize {
        let tt = t[usize::from(g)][m];
        if m == 2 {
            2 * tt.layer_words()
        } else {
            tt.layer_words()
        }
    };
    hdrs + glob.iter().map(|&g| per(g)).sum::<usize>()
}

/// The global RoPE's facts read off the checkpoint's config: per kind
/// (sliding, global) theta, head_dim and the dims it turns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ropes {
    pub theta: [f64; 2],
    pub rotary: [usize; 2],
}

impl Ropes {
    /// `rope_parameters` of a Gemma 4 config (or its `text_config`):
    /// sliding over every pair, global (`proportional`) over
    /// `partial_rotary_factor` of its head.
    pub fn read(ckpt: &Ckpt, hd: [usize; 2]) -> R<Ropes> {
        let c = ckpt.config();
        let rp = c
            .get("rope_parameters")
            .or_else(|| c.get("text_config").and_then(|t| t.get("rope_parameters")))
            .ok_or("Gemma 4 checkpoint config without rope_parameters")?;
        let theta = |k: &str| -> R<f64> {
            rp[k]["rope_theta"]
                .as_f64()
                .ok_or_else(|| format!("rope_parameters.{k}.rope_theta"))
        };
        let part = rp["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap_or(1.0);
        Ok(Ropes {
            theta: [theta("sliding_attention")?, theta("full_attention")?],
            rotary: [hd[0], (part * hd[1] as f64) as usize],
        })
    }

    /// Kind `g`'s inverse frequency of pair `i` (< head_dim / 2), as
    /// transformers' (proportional: zero past the turned pairs).
    #[must_use]
    pub fn inv_freq(&self, g: usize, hd: usize, i: usize) -> f64 {
        if 2 * i >= self.rotary[g] {
            return 0.0;
        }
        self.theta[g].powf(-((2 * i) as f64) / hd as f64)
    }
}

/// Column px's q/k dims of its kv head (copy) for kind `s`: a DH run from
/// each RoPE half.
fn qk_dims(px: usize, dc: usize, s: KindShape) -> Vec<usize> {
    let (dh, d, j) = (dc / 2, s.gs * dc, px % s.gs);
    (j * dh..(j + 1) * dh)
        .chain((j * dh..(j + 1) * dh).map(|i| d / 2 + i))
        .collect()
}

/// Column px's v (and o) dims: Dc consecutive dims of the head.
fn v_dims(px: usize, dc: usize, s: KindShape) -> std::ops::Range<usize> {
    let j = px % s.gs;
    j * dc..(j + 1) * dc
}

/// A unit's per-layer facts.
struct Lay {
    l: usize,
    s: KindShape,
}

/// One unit symbol's u16 words for PE (py, px) from each layer's tile rows
/// and columns (`cols[l]`: one or several column sets, wgu's gate and up):
/// the distinct tiles' headers (sliding's first), then the layers' words.
fn assemble<'q>(
    ts: &[[Tile; 5]; 2],
    m: usize,
    buf: usize,
    lays: &[Lay],
    rows: &dyn Fn(KindShape) -> Vec<Option<usize>>,
    cols: &dyn Fn(usize) -> Vec<Vec<Col<'q>>>,
    out: &mut Vec<u32>,
) -> R<()> {
    let mut hdr: [Option<Vec<u32>>; 2] = [None, None];
    let mut words = Vec::new();
    for (li, ly) in lays.iter().enumerate() {
        let gi = usize::from(ly.s.glob);
        let t = ts[gi][m];
        let mut b = Vec::new();
        blob_in(t, buf, &rows(ly.s), &cols(li), &mut b)?;
        let h = t.header_in(buf);
        match &hdr[gi] {
            Some(prev) if prev[..] != b[..h] => {
                return Err(format!(
                    "u4g12: layer {}'s header differs from its kind's",
                    ly.l
                ));
            }
            Some(_) => {}
            None => hdr[gi] = Some(b[..h].to_vec()),
        }
        words.extend_from_slice(&b[h..]);
    }
    let both = hdr[0].is_some() && hdr[1].is_some();
    match (&hdr[0], &hdr[1]) {
        (Some(s), Some(g)) if both && ts[0][m] != ts[1][m] => {
            out.extend_from_slice(s);
            out.extend_from_slice(g);
        }
        (Some(s), _) => out.extend_from_slice(s),
        (None, Some(g)) => out.extend_from_slice(g),
        (None, None) => return Err("u4g12: a unit without layers".into()),
    }
    out.extend_from_slice(&words);
    Ok(())
}

/// Every unit's weights, norms, RoPE tables and zeroed KV.
pub(crate) fn place_layers(
    lay: &ProgramSpec,
    u: &Units,
    g: &G12,
    ckpt: &Ckpt,
    dev: &mut dyn Device,
) -> R<()> {
    if u.nolayer {
        return Ok(());
    }
    for v in u.units(lay) {
        place_unit(lay, g, ckpt, &v, dev)?;
    }
    Ok(())
}

/// The unit's most query heads a layer (its lane layout's) for its kinds.
#[must_use]
pub fn unit_qm(ks: &[KindShape; 2], glob: &[bool]) -> usize {
    if glob.iter().any(|g| *g) && ks[1].qh > ks[0].qh {
        ks[1].qh
    } else {
        ks[0].qh
    }
}

#[allow(clippy::too_many_lines)]
fn place_unit(lay: &ProgramSpec, g: &G12, ckpt: &Ckpt, v: &Unit, dev: &mut dyn Device) -> R<()> {
    let k = v.k;
    let (hc, dc, ic, dh) = (k.hc, k.dc, k.ic, k.dc / 2);
    let hid = lay.hidden();
    let ks = kinds(lay.qh(), k.gs, g);
    let ts = [tiles(k, ks[0]), tiles(k, ks[1])];
    let buf = g.w4buf;
    let lays: Vec<Lay> = v
        .layers
        .clone()
        .map(|l| Lay {
            l,
            s: ks[usize::from(g.global(l))],
        })
        .collect();
    let glob: Vec<bool> = lays.iter().map(|y| y.s.glob).collect();
    let nl = lays.len();
    let qm = unit_qm(&ks, &glob);
    let nh = qm + 1;
    let inter = k.px * ic;
    let at = (v.x0, v.y0);
    let name = |l: usize, w: &str| format!("layers.{l}.{w}");
    let hd = |s: KindShape| s.gs * dc;
    let nkv = |s: KindShape| if s.glob { 1 } else { k.px / s.gs };
    let nq = |s: KindShape| (k.px / s.gs) * s.qh;
    let dim = |py: usize, kk: usize| Some(py * hc + kk).filter(|&x| x < hid);
    let hidden_rows = |py: usize| -> Vec<Option<usize>> { (0..hc).map(|kk| dim(py, kk)).collect() };
    // Column px's kv head (copy) index into k_proj/v_proj, and its first query head.
    let kvh = |s: KindShape, px: usize| if s.glob { 0 } else { px / s.gs };
    let q0 = |s: KindShape, px: usize| (px / s.gs) * s.qh;
    let words = |m: usize| symbol_words(k, &ks, &glob, buf, m);

    // QKV: [q heads' then k's first-half dims | second halves | v], fused.
    let wq: Vec<Q4> = lays
        .iter()
        .map(|y| Q4::of(ckpt, &name(y.l, "q"), nq(y.s) * hd(y.s), hid))
        .collect::<R<_>>()?;
    let wk: Vec<Q4> = lays
        .iter()
        .map(|y| Q4::of(ckpt, &name(y.l, "k"), nkv(y.s) * hd(y.s), hid))
        .collect::<R<_>>()?;
    // A global layer's v is its raw k (attention_k_eq_v).
    let wv: Vec<Option<Q4>> = lays
        .iter()
        .map(|y| {
            if y.s.glob {
                Ok(None)
            } else {
                Q4::of(ckpt, &name(y.l, "v"), nkv(y.s) * hd(y.s), hid).map(Some)
            }
        })
        .collect::<R<_>>()?;
    let err = std::cell::RefCell::new(None::<String>);
    let put = |dev: &mut dyn Device,
               sym: &str,
               m: usize,
               f: &dyn Fn(usize, usize, &mut Vec<u32>) -> R<()>|
     -> R<()> {
        let n = words(m);
        put_pe16(k, dev, sym, at, n, |py, px, out| {
            let start = out.len();
            if let Err(e) = f(py, px, out) {
                err.borrow_mut().get_or_insert(format!(
                    "u4g12 {sym} at ({px}, {py}) of unit {}: {e}",
                    v.idx
                ));
            }
            if out.len() - start != n {
                err.borrow_mut().get_or_insert(format!(
                    "u4g12 {sym}: {} words, the kernel's {n}",
                    out.len() - start
                ));
            }
            out.truncate(start + n);
            out.resize(start + n, 0);
        })?;
        err.borrow_mut().take().map_or(Ok(()), Err)
    };
    put(dev, "wqa", 0, &|py, px, out| {
        assemble(
            &ts,
            0,
            buf,
            &lays,
            &|_| hidden_rows(py),
            &|li| {
                let s = lays[li].s;
                let (dl, kv, q0) = (qk_dims(px, dc, s), kvh(s, px), q0(s, px));
                let d = hd(s);
                let half = |h0: usize| -> Vec<Col<'_>> {
                    (0..=s.qh)
                        .flat_map(|h| (0..dh).map(move |i| (h, i)))
                        .map(|(h, i)| {
                            let dd = dl[h0 + i];
                            Some(if h < s.qh {
                                (&wq[li], (q0 + h) * d + dd)
                            } else {
                                (&wk[li], kv * d + dd)
                            })
                        })
                        .collect()
                };
                let mut c = half(0);
                c.extend(half(dh));
                let src = wv[li].as_ref().unwrap_or(&wk[li]);
                c.extend(v_dims(px, dc, s).map(|dd| Some((src, kv * d + dd))));
                vec![c]
            },
            out,
        )
    })?;
    drop((wq, wk, wv));
    let wo: Vec<Q4> = lays
        .iter()
        .map(|y| Q4::of(ckpt, &name(y.l, "o"), hid, nq(y.s) * hd(y.s)))
        .collect::<R<_>>()?;
    put(dev, "wo", 1, &|py, px, out| {
        assemble(
            &ts,
            1,
            buf,
            &lays,
            &|s| {
                let (q0, d) = (q0(s, px), hd(s));
                (0..s.qh)
                    .flat_map(|h| v_dims(px, dc, s).map(move |dd| Some((q0 + h) * d + dd)))
                    .collect()
            },
            &|li| {
                vec![
                    (0..hc)
                        .map(|kk| dim(py, kk).map(|o| (&wo[li], o)))
                        .collect(),
                ]
            },
            out,
        )
    })?;
    drop(wo);
    let wg: Vec<Q4> = lays
        .iter()
        .map(|y| Q4::of(ckpt, &name(y.l, "gate"), inter, hid))
        .collect::<R<_>>()?;
    let wu: Vec<Q4> = lays
        .iter()
        .map(|y| Q4::of(ckpt, &name(y.l, "up"), inter, hid))
        .collect::<R<_>>()?;
    put(dev, "wgu", 2, &|py, px, out| {
        assemble(
            &ts,
            2,
            buf,
            &lays,
            &|_| hidden_rows(py),
            &|li| {
                [&wg[li], &wu[li]]
                    .into_iter()
                    .map(|w| (0..ic).map(|i| Some((w, px * ic + i))).collect())
                    .collect()
            },
            out,
        )
    })?;
    drop((wg, wu));
    let wd: Vec<Q4> = lays
        .iter()
        .map(|y| Q4::of(ckpt, &name(y.l, "down"), hid, inter))
        .collect::<R<_>>()?;
    put(dev, "wd", 4, &|py, px, out| {
        assemble(
            &ts,
            4,
            buf,
            &lays,
            &|_| (0..ic).map(|i| Some(px * ic + i)).collect(),
            &|li| {
                vec![
                    (0..hc)
                        .map(|kk| dim(py, kk).map(|o| (&wd[li], o)))
                        .collect(),
                ]
            },
            out,
        )
    })?;
    drop(wd);

    // q/k norms [layer][half][QM + 1][DH]: the layer's query heads, its k at
    // slot qh, zeros past it.
    let vec_of = |l: usize, leaf: &str, n: usize| -> R<Vec<f32>> {
        let t = ckpt.hf(&format!("layers.{l}.{leaf}"))?;
        if t.data.len() != n {
            return Err(format!(
                "layers.{l}.{leaf} is {:?}; the program wants [{n}]",
                t.shape
            ));
        }
        Ok(t.data)
    };
    let qn: Vec<Vec<f32>> = lays
        .iter()
        .map(|y| vec_of(y.l, "self_attn.q_norm.weight", hd(y.s)))
        .collect::<R<_>>()?;
    let kn: Vec<Vec<f32>> = lays
        .iter()
        .map(|y| vec_of(y.l, "self_attn.k_norm.weight", hd(y.s)))
        .collect::<R<_>>()?;
    put32(k, dev, "qkn_w", at, nl * 2 * nh * dh, &|_, px, out| {
        for (li, y) in lays.iter().enumerate() {
            let dl = qk_dims(px, dc, y.s);
            for half in [0, dh] {
                for h in 0..nh {
                    for i in 0..dh {
                        let dd = dl[half + i];
                        let w = if h < y.s.qh {
                            qn[li][dd]
                        } else if h == y.s.qh {
                            kn[li][dd]
                        } else {
                            0.0
                        };
                        out.push(w.to_bits());
                    }
                }
            }
        }
    })?;
    // RoPE [kind present][RB][DH]: cos/sin(2^lv f_i), the sliding kind's first.
    let rope = Ropes::read(ckpt, [hd(ks[0]), hd(ks[1])])?;
    let present: Vec<KindShape> = ks
        .iter()
        .copied()
        .filter(|s| glob.contains(&s.glob))
        .collect();
    for (sym, sin) in [("rot_c", false), ("rot_s", true)] {
        put32(k, dev, sym, at, present.len() * RB * dh, &|_, px, out| {
            for s in &present {
                let dl = qk_dims(px, dc, *s);
                for lv in 0..RB {
                    for &i in &dl[..dh] {
                        let ang =
                            f64::from(1u32 << lv) * rope.inv_freq(usize::from(s.glob), hd(*s), i);
                        out.push((if sin { ang.sin() } else { ang.cos() } as f32).to_bits());
                    }
                }
            }
        })?;
    }
    // gamma1 [layer][Hc] (input norm), gamma2 [layer][pre-ffw | post-attn | post-ffw | scalar].
    let g1: Vec<Vec<f32>> = lays
        .iter()
        .map(|y| vec_of(y.l, "input_layernorm.weight", hid))
        .collect::<R<_>>()?;
    put32(k, dev, "gamma1", at, nl * hc, &|py, _, out| {
        for g in &g1 {
            out.extend((0..hc).map(|kk| dim(py, kk).map_or(0, |x| g[x].to_bits())));
        }
    })?;
    let mut g2 = Vec::new();
    for y in &lays {
        let s = vec_of(y.l, "layer_scalar", 1)?[0];
        g2.push((
            vec_of(y.l, "pre_feedforward_layernorm.weight", hid)?,
            vec_of(y.l, "post_attention_layernorm.weight", hid)?,
            vec_of(y.l, "post_feedforward_layernorm.weight", hid)?,
            s,
        ));
    }
    put32(k, dev, "gamma2", at, nl * (3 * hc + 1), &|py, _, out| {
        for (pre, post_a, post_f, s) in &g2 {
            for w in [pre, post_a, post_f] {
                out.extend((0..hc).map(|kk| dim(py, kk).map_or(0, |x| w[x].to_bits())));
            }
            out.push(s.to_bits());
        }
    })?;
    // The pool, zeroed.
    let shape = lay
        .paged()
        .ok_or("serve8b_region without its paged shape")?;
    let kv = nl * shape.blocks * dc * shape.page_size;
    for sym in ["kc", "vc"] {
        put_pe16(k, dev, sym, at, kv, |_, _, out| {
            out.extend(std::iter::repeat_n(0, kv))
        })?;
    }
    Ok(())
}

/// A unit's f32 per-PE blocks (`n` words each).
fn put32(
    k: Kernel,
    dev: &mut dyn Device,
    sym: &str,
    at: (usize, usize),
    n: usize,
    f: &dyn Fn(usize, usize, &mut Vec<u32>),
) -> R<()> {
    let mut words = Vec::with_capacity(k.px * k.py * n);
    for py in 0..k.py {
        for px in 0..k.px {
            let start = words.len();
            f(py, px, &mut words);
            if words.len() - start != n {
                return Err(format!(
                    "u4g12 {sym}: {} words a PE, the kernel's {n}",
                    words.len() - start
                ));
            }
        }
    }
    dev.put(sym, rect(at.0, at.1, k.px, k.py), n, &words)
}

/// A unit PE's [`RCFG`] route words: the sliding kind's 11 (X, Y, group
/// broadcast) then the global kind's X (5) and group broadcast.
#[must_use]
pub fn unit_route_words(
    rb: &[u32],
    px_n: usize,
    py_n: usize,
    gs: usize,
    gsg: usize,
    gy: usize,
) -> Vec<u32> {
    let s = route_words(rb, px_n, py_n, gs, gy);
    let g = route_words(rb, px_n, py_n, gsg, gy);
    let mut out = Vec::with_capacity(px_n * py_n * RCFG);
    for pe in 0..px_n * py_n {
        out.extend_from_slice(&s[pe * 11..pe * 11 + 11]);
        out.extend_from_slice(&g[pe * 11..pe * 11 + 5]);
        out.push(g[pe * 11 + 10]);
    }
    out
}

/// Every unit's `rcfg` (both kinds' words).
pub(crate) fn put_rcfg(
    lay: &ProgramSpec,
    u: &Units,
    g: &G12,
    dev: &mut dyn Device,
    rb: &[u32],
) -> R<()> {
    for v in u.units(lay) {
        let ks = kinds(lay.qh(), v.k.gs, g);
        let gy = if v.kind == Kind::Grid { u.gy } else { u.rgy };
        let words = unit_route_words(rb, v.k.px, v.k.py, ks[0].gs, ks[1].gs, gy);
        dev.put("rcfg", rect(v.x0, v.y0, v.k.px, v.k.py), RCFG, &words)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g12() -> G12 {
        G12 {
            kinds: (0..48).filter(|l| l % 6 == 5).map(|l| 1u64 << l).sum(),
            qhg: 4,
            gsg: 32,
            win: 1024,
            w4buf: 512,
            esb: (3840f32).sqrt().to_bits(),
        }
    }

    /// The host's symbol sizes equal the kernel's, as linked at Gemma-4-12B's
    /// shapes (link probe scratch/g12/lp/g12d, W4BUF 512: `nm` of each image,
    /// bytes / 2 less the pad): grids 128 x 96 (Hc 40, Ic 120), regions 128 x
    /// 240 (RHc 16) of two layers; sliding Qh 2 Gs 16, global QhG 4 GsG 32, Dc 16.
    #[test]
    fn symbol_words_match_the_linked_kernel() {
        let g = g12();
        let grid = Kernel {
            px: 128,
            py: 96,
            hc: 40,
            dc: 16,
            gs: 16,
            ic: 120,
        };
        let region = Kernel {
            px: 128,
            py: 240,
            hc: 16,
            dc: 16,
            gs: 16,
            ic: 120,
        };
        let ks = kinds(2, 16, &g);
        for (k, glob, want) in [
            (grid, vec![false], [944, 510, 3434, 1528]),
            (grid, vec![true], [1392, 1022, 3434, 1528]),
            (region, vec![false, false], [794, 422, 2914, 1238]),
            (region, vec![true, false], [1004, 652, 2914, 1238]),
        ] {
            let got: Vec<usize> = [0, 1, 2, 4]
                .iter()
                .map(|&m| symbol_words(k, &ks, &glob, 512, m))
                .collect();
            assert_eq!(got, want, "{k:?} {glob:?}");
        }
    }

    /// The rcfg words: the sliding kind's 11 as route_words gives them, then
    /// the global kind's X and group broadcast.
    #[test]
    fn route_words_hold_both_kinds() {
        let rb: Vec<u32> = (0..10).map(|i| 1 << i).collect();
        let w = unit_route_words(&rb, 128, 16, 16, 32, 16);
        let (s, g) = (
            route_words(&rb, 128, 16, 16, 16),
            route_words(&rb, 128, 16, 32, 16),
        );
        assert_eq!(w.len(), 128 * 16 * RCFG);
        for pe in [0, 17, 128 * 16 - 1] {
            assert_eq!(&w[pe * RCFG..pe * RCFG + 11], &s[pe * 11..pe * 11 + 11]);
            assert_eq!(&w[pe * RCFG + 11..pe * RCFG + 16], &g[pe * 11..pe * 11 + 5]);
            assert_eq!(w[pe * RCFG + 16], g[pe * 11 + 10]);
            assert_eq!(
                &w[pe * RCFG + 5..pe * RCFG + 10],
                &g[pe * 11 + 5..pe * 11 + 10],
                "Y words are the kinds' same"
            );
        }
    }
}
