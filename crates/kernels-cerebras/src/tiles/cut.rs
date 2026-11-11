//! The one tiler: a [`Spec`]'s arrays made from the params, each written to `<dir>/<name>.npy`
//! as soon as it is made.
//!
//! Rounding is numpy's (f16 to nearest even), so the arrays are the earlier Python preps' byte for
//! byte, except one: Gemma's per-layer inputs are computed in f64 throughout and rounded once to
//! f16, where the prep summed its projection in f32 in its BLAS build's own order. Those tables are
//! the reference now: they are the f64 values' nearest f16, and about 0.1% of their values differ
//! from the prep's (where its f32 sums were off).

use std::collections::VecDeque;
use std::sync::Arc;

use super::spec::{
    Bank, Elem, ExpertScales, Experts, File, Fill, Gather, Grid, Head, Map, Part, Ple, Rope, Spec,
};
use super::{Out, Params, R, exp2i, f16};

/// Threads a cut runs on (the shared host's BLAS cap).
const THREADS: usize = 8;

/// Bytes of widened params kept between arrays (a unit's weights feed several).
const KEEP: usize = 16 << 30;

/// A param (or params stacked) widened to f32: `[rows][width]`.
struct Mat {
    data: Vec<f32>,
    width: usize,
}

/// The params last widened, newest last.
struct Cache<'a> {
    params: &'a dyn Params,
    mats: VecDeque<(String, Arc<Mat>)>,
    bytes: usize,
}

impl<'a> Cache<'a> {
    fn new(params: &'a dyn Params) -> Self {
        Cache {
            params,
            mats: VecDeque::new(),
            bytes: 0,
        }
    }

    fn get(&mut self, src: &[String]) -> R<Arc<Mat>> {
        let key = src.join("+");
        if let Some((_, m)) = self.mats.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(m));
        }
        let mut data = Vec::new();
        let mut width = None;
        for name in src {
            let p = self
                .params
                .get(name)
                .ok_or_else(|| format!("no param {name}"))?;
            // a row: a matrix's (its dims past the first); a bank of matrices (three dims or
            // more) its last dim's, rows running over the leading ones
            let w = if p.shape.len() >= 3 {
                p.shape.last().copied().unwrap_or(1) as usize
            } else {
                p.shape.iter().skip(1).product::<u64>() as usize
            };
            if *width.get_or_insert(w) != w {
                return Err(format!("{key}: stacked params of different widths"));
            }
            data.extend(p.f32()?);
        }
        let m = Arc::new(Mat {
            data,
            width: width.unwrap_or(1),
        });
        self.bytes += m.data.len() * 4;
        self.mats.push_back((key, Arc::clone(&m)));
        while self.bytes > KEEP && self.mats.len() > 1 {
            if let Some((_, old)) = self.mats.pop_front() {
                self.bytes -= old.data.len() * 4;
            }
        }
        Ok(m)
    }
}

/// An array's element store.
trait Word: Copy + Default + Send + Sync + super::Store {
    fn of(elem: Elem, v: f32) -> Self;
    fn raw(bits: u16) -> Self;
}

impl Word for u16 {
    fn of(elem: Elem, v: f32) -> u16 {
        match elem {
            Elem::F16 => f16(v),
            Elem::Bf16 => (v.to_bits() >> 16) as u16,
            Elem::U16 => v as u16,
            Elem::F32 => unreachable!("an f32 array stores f32"),
        }
    }
    fn raw(bits: u16) -> u16 {
        bits
    }
}

impl Word for f32 {
    fn of(_: Elem, v: f32) -> f32 {
        v
    }
    fn raw(bits: u16) -> f32 {
        f32::from(bits)
    }
}

/// Makes every array of `spec` from `params` into `out`.
pub fn cut(spec: &Spec, params: &dyn Params, out: &Out) -> R<()> {
    let mut cache = Cache::new(params);
    for f in &spec.files {
        match f.elem {
            Elem::F32 => {
                let a = make::<f32>(f, &mut cache)?;
                out.save(&f.name, &f.shape, &a)?;
            }
            _ => {
                let a = make::<u16>(f, &mut cache)?;
                out.save(&f.name, &f.shape, &a)?;
            }
        }
    }
    Ok(())
}

fn make<T: Word>(f: &File, cache: &mut Cache) -> R<Vec<T>> {
    let mut a = vec![T::default(); f.len()];
    let ctx = |e: String| format!("{}: {e}", f.name);
    for fill in &f.fills {
        match fill {
            Fill::Grid(g) => grid(f, g, cache, &mut a).map_err(ctx)?,
            Fill::Gather(g) => gather(f.elem, g, cache, &mut a).map_err(ctx)?,
            Fill::Copy { at, src, map } => {
                let m = cache.get(std::slice::from_ref(src)).map_err(ctx)?;
                let dst = a
                    .get_mut(*at..*at + m.data.len())
                    .ok_or_else(|| ctx(format!("{src} runs past the array")))?;
                for (d, &v) in dst.iter_mut().zip(&m.data) {
                    *d = T::of(f.elem, apply(*map, v));
                }
            }
            Fill::Const { at, value } => {
                for &i in at {
                    *a.get_mut(i)
                        .ok_or_else(|| ctx(format!("{i} past the array")))? = T::of(f.elem, *value);
                }
            }
            Fill::Rope(r) => rope(f.elem, r, &mut a).map_err(ctx)?,
            Fill::Head(h) => head(f.elem, h, cache, &mut a).map_err(ctx)?,
            Fill::Ple(p) => ple(f.elem, p, cache, &mut a).map_err(ctx)?,
            Fill::Experts(x) => experts(x, cache.params, &mut a).map_err(ctx)?,
            Fill::ExpertScales(x) => expert_scales(f.elem, x, cache.params, &mut a).map_err(ctx)?,
        }
    }
    Ok(a)
}

fn apply(map: Map, v: f32) -> f32 {
    match map {
        Map::Id => v,
        Map::PlusOne => 1.0 + v,
        Map::Div(d) => v / d,
        Map::PlusOneDiv(d) => (1.0 + v) / d,
    }
}

/// `n` items over the threads: `f(i, chunk)` on `a`'s consecutive chunks of `per` words.
fn parallel<T: Send>(
    a: &mut [T],
    per: usize,
    f: impl Fn(usize, &mut [T]) -> R<()> + Sync,
) -> R<()> {
    if per == 0 {
        return Ok(());
    }
    let chunks: Vec<(usize, &mut [T])> = a.chunks_mut(per).enumerate().collect();
    let n = chunks.len();
    let mut lanes: Vec<Vec<(usize, &mut [T])>> = (0..THREADS).map(|_| Vec::new()).collect();
    for (i, c) in chunks {
        lanes[i * THREADS / n.max(1)].push((i, c));
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = lanes
            .into_iter()
            .map(|lane| {
                let f = &f;
                s.spawn(move || {
                    for (i, c) in lane {
                        f(i, c)?;
                    }
                    Ok::<(), String>(())
                })
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().map_err(|_| "a cut thread panicked".to_string())?)
    })
}

fn grid<T: Word>(f: &File, g: &Grid, cache: &mut Cache, a: &mut [T]) -> R<()> {
    let (nr, nc) = (g.rows.len(), f.shape.get(1).copied().unwrap_or(0));
    let len = g.rows.first().map_or(0, |r| r.1);
    if f.shape.len() != 3 || f.shape[0] != nr || g.rows.iter().any(|r| r.1 != len) {
        return Err(format!(
            "a grid of {nr} rows of {len} dims on {:?}",
            f.shape
        ));
    }
    let width = f.shape[2];
    let mut mats = Vec::new();
    let mut off = 0;
    for p in &g.parts {
        let (src, idx) = match p {
            Part::In { src, outs } => (src, outs),
            Part::Out { src, ins } => (src, ins),
        };
        if idx.len() != nc || idx.iter().any(|i| i.len() != idx[0].len()) {
            return Err(format!("{}: not {nc} columns of one length", src.join("+")));
        }
        mats.push((cache.get(src)?, off));
        off += len * idx[0].len();
    }
    if off != width {
        return Err(format!("parts of {off} words a PE, the array's {width}"));
    }
    let elem = f.elem;
    parallel(a, nc * width, |r, row| {
        for (p, (m, off)) in g.parts.iter().zip(&mats) {
            let (d0, h) = (g.rows[r].0, g.hidden);
            match p {
                Part::In { outs, .. } => {
                    for (c, cols) in outs.iter().enumerate() {
                        let n = cols.len();
                        let dst = &mut row[c * width + off..c * width + off + len * n];
                        for i in 0..len {
                            let d = d0 + i;
                            for (k, &o) in cols.iter().enumerate() {
                                let v = if o < 0 || d >= h {
                                    0.0
                                } else {
                                    m.data[o as usize * m.width + d]
                                };
                                dst[i * n + k] = T::of(elem, v);
                            }
                        }
                    }
                }
                Part::Out { ins, .. } => {
                    for (c, cols) in ins.iter().enumerate() {
                        let n = cols.len();
                        let dst = &mut row[c * width + off..c * width + off + len * n];
                        for (k, &o) in cols.iter().enumerate() {
                            for i in 0..len {
                                let d = d0 + i;
                                let v = if o < 0 || d >= h {
                                    0.0
                                } else {
                                    m.data[d * m.width + o as usize]
                                };
                                dst[k * len + i] = T::of(elem, v);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

fn gather<T: Word>(elem: Elem, g: &Gather, cache: &mut Cache, a: &mut [T]) -> R<()> {
    let m = cache.get(std::slice::from_ref(&g.src))?;
    let dst = a
        .get_mut(g.at..g.at + g.idx.len())
        .ok_or_else(|| format!("{} runs past the array", g.src))?;
    for (d, &i) in dst.iter_mut().zip(&g.idx) {
        let v = if i < 0 {
            0.0
        } else {
            apply(
                g.map,
                *m.data
                    .get(i as usize)
                    .ok_or_else(|| format!("{}[{i}] out of range", g.src))?,
            )
        };
        *d = T::of(elem, v);
    }
    Ok(())
}

fn rope<T: Word>(elem: Elem, r: &Rope, a: &mut [T]) -> R<()> {
    let mut at = 0;
    for col in &r.freqs {
        for kind in col {
            let np = kind.len();
            for lv in 0..r.levels {
                let step = r.radix.pow((r.levels - 1 - lv) as u32);
                let sc = if lv == 0 { r.scale0 } else { 1.0 };
                for d in 0..r.radix {
                    for (j, &f) in kind.iter().enumerate() {
                        let ang = (d * step) as f64 * f;
                        a[at + j] = T::of(elem, (ang.cos() * sc) as f32);
                        a[at + np + j] = T::of(elem, (ang.sin() * sc) as f32);
                    }
                    at += 2 * np;
                }
            }
        }
    }
    if at != a.len() {
        return Err(format!(
            "RoPE tables of {at} values, the array's {}",
            a.len()
        ));
    }
    Ok(())
}

fn head<T: Word>(elem: Elem, h: &Head, cache: &mut Cache, a: &mut [T]) -> R<()> {
    let m = cache.get(std::slice::from_ref(&h.src))?;
    let nd = h.dims.len();
    let span = h.dims.first().map_or(0, Vec::len);
    let tile = span * h.per;
    if a.len() != h.groups * nd * tile {
        return Err(format!(
            "head tiles {} x {nd} x {tile}, the array's {}",
            h.groups,
            a.len()
        ));
    }
    let (v, w) = (h.vocab, m.width);
    let scale = h.scale;
    let inners = if h.dims_major { h.groups } else { nd };
    parallel(a, inners * tile, |outer, blk| {
        for inner in 0..inners {
            let (g, j) = if h.dims_major {
                (inner, outer)
            } else {
                (outer, inner)
            };
            let dst = &mut blk[inner * tile..(inner + 1) * tile];
            for (k, &d) in h.dims[j].iter().enumerate() {
                for row in 0..h.per {
                    let id = g * h.per + row;
                    let src = if id < v {
                        Some(id)
                    } else if h.copy_last {
                        Some(v - 1)
                    } else {
                        None
                    };
                    let x = match (src, d) {
                        (Some(s), d) if d >= 0 => m.data[s * w + d as usize] * scale,
                        _ => 0.0,
                    };
                    let at = if h.row_major {
                        row * span + k
                    } else {
                        k * h.per + row
                    };
                    dst[at] = T::of(elem, x);
                }
            }
        }
        Ok(())
    })
}

/// `a . b` accumulated in f64.
fn dot(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += f64::from(x[k]) * f64::from(y[k]);
        }
    }
    let mut s = acc.iter().sum::<f64>();
    for (x, y) in ra.iter().zip(rb) {
        s += f64::from(*x) * f64::from(*y);
    }
    s
}

fn ple<T: Word>(elem: Elem, p: &Ple, cache: &mut Cache, a: &mut [T]) -> R<()> {
    let e = cache.get(std::slice::from_ref(&p.embed))?;
    let w = cache.get(std::slice::from_ref(&p.proj))?;
    let n = cache.get(std::slice::from_ref(&p.norm))?;
    let tables = p
        .layers
        .iter()
        .map(|&l| {
            p.tables
                .get(l)
                .ok_or_else(|| format!("no per-layer table {l}"))
                .and_then(|t| cache.get(std::slice::from_ref(t)))
        })
        .collect::<R<Vec<_>>>()?;
    let (h, pw, nl) = (e.width, p.width, p.layers.len());
    let sp = (pw as f64).sqrt();
    let half = 0.5f64.sqrt();
    let eps = f64::from(p.eps);
    let per = pw * p.nt * nl;
    if a.len() != (p.rows.1 - p.rows.0) * per {
        return Err(format!(
            "{} rows of {per}, the array's {}",
            p.rows.1 - p.rows.0,
            a.len()
        ));
    }
    // every value in f64 (the embedding's sqrt(H) and the projection's 1 / sqrt(H) cancel),
    // rounded once to the array's type
    parallel(a, per, |k, blk| {
        let mut pr = vec![0.0f64; pw];
        for j in 0..p.nt {
            let t = (p.rows.0 + k) * p.nt + j;
            if t >= p.vocab {
                break;
            }
            let x = &e.data[t * h..(t + 1) * h];
            for (i, &l) in p.layers.iter().enumerate() {
                for (o, v) in pr.iter_mut().enumerate() {
                    *v = dot(x, &w.data[(l * pw + o) * h..(l * pw + o + 1) * h]);
                }
                let ms = pr.iter().map(|v| v * v).sum::<f64>() / pw as f64;
                let s = (ms + eps).sqrt();
                let tb = &tables[i].data[t * pw..(t + 1) * pw];
                for o in 0..pw {
                    let v = (pr[o] / s * f64::from(n.data[o]) + f64::from(tb[o]) * sp) * half;
                    blk[o * p.nt * nl + j * nl + i] = T::of(elem, narrow(v));
                }
            }
        }
        Ok(())
    })
}

/// `x` as the f32 that rounds on to any narrower type as `x` itself would: rounded to odd (the
/// truncation, its last bit set where it was inexact), so f64 -> f32 -> f16 rounds once.
fn narrow(x: f64) -> f32 {
    let t = x as f32;
    if f64::from(t) == x || !t.is_finite() {
        return t;
    }
    // toward zero, then the sticky bit
    let t = if f64::from(t).abs() > x.abs() {
        f32::from_bits(t.to_bits() - 1)
    } else {
        t
    };
    f32::from_bits(t.to_bits() | 1)
}

/// MXFP4's code values (low nibble first in a byte).
const FP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// One expert's leg: its codes `[rows][G][16]` and scales `[rows][G]`.
struct Leg<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    groups: usize,
}

impl Leg<'_> {
    fn of<'a>(
        params: &'a dyn Params,
        codes: &str,
        scales: &str,
        e: usize,
        experts: usize,
    ) -> R<Leg<'a>> {
        let c = params
            .get(codes)
            .ok_or_else(|| format!("no param {codes}"))?;
        let s = params
            .get(scales)
            .ok_or_else(|| format!("no param {scales}"))?;
        let (rows, groups) = (c.shape[1] as usize, c.shape[2] as usize);
        let (cb, sb) = (c.bytes.len() / experts, s.bytes.len() / experts);
        if cb != rows * groups * 16 || sb != rows * groups {
            return Err(format!(
                "{codes}: not {experts} experts of [{rows}][{groups}][16] codes"
            ));
        }
        Ok(Leg {
            codes: &c.bytes[e * cb..(e + 1) * cb],
            scales: &s.bytes[e * sb..(e + 1) * sb],
            rows,
            groups,
        })
    }

    /// The leg's largest scale over the groups that hold a nonzero code (0: none does).
    fn top(&self) -> i32 {
        let mut m = 0u8;
        for (g, blk) in self.codes.as_chunks::<16>().0.iter().enumerate() {
            if blk.iter().any(|b| b & 0x77 != 0) {
                m = m.max(self.scales[g]);
            }
        }
        i32::from(m)
    }

    /// Row `r`'s values (HF's dequant: code value times `2^(scale - 127)`).
    fn row(&self, r: usize, out: &mut [f32]) {
        for g in 0..self.groups {
            let s = exp2i(i32::from(self.scales[r * self.groups + g]) - 127);
            let blk = &self.codes[(r * self.groups + g) * 16..(r * self.groups + g + 1) * 16];
            for (i, b) in blk.iter().enumerate() {
                out[g * 32 + 2 * i] = FP4[(b & 0x0f) as usize] * s;
                out[g * 32 + 2 * i + 1] = FP4[(b >> 4) as usize] * s;
            }
        }
    }

    /// Every value times `2^-p` as E5M2 (an f16's high byte), exact and normal, or why not.
    fn e5m2(&self, p: i32, what: &str) -> R<Vec<u8>> {
        let width = self.groups * 32;
        let (down, up) = (exp2i(-p), exp2i(p));
        let mut out = vec![0u8; self.rows * width];
        let mut row = vec![0.0f32; width];
        let (mut bad, mut sub) = (0usize, 0usize);
        for r in 0..self.rows {
            self.row(r, &mut row);
            for (k, &t) in row.iter().enumerate() {
                let hb = f16(t * down);
                let back = half_to_f32(hb & 0xff00) * up;
                bad += usize::from(back != t);
                let mag = hb & 0x7fff;
                sub += usize::from(mag != 0 && mag < 0x0400);
                out[r * width + k] = (hb >> 8) as u8;
            }
        }
        if bad > 0 || sub > 0 {
            return Err(format!(
                "E5M2 check: {what}: {bad} weights not exact, {sub} subnormal (P {p})"
            ));
        }
        Ok(out)
    }
}

/// An f16's value.
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    sign * match exp {
        0 => man * exp2i(-24),
        0x1f if man == 0.0 => f32::INFINITY,
        0x1f => f32::NAN,
        e => (1.0 + man / 1024.0) * exp2i(e - 15),
    }
}

/// Each layer's experts' (gate | up, down) exponents P.
fn exponents(params: &dyn Params, banks: &[Bank], experts: usize) -> R<Vec<(i32, i32)>> {
    let mut ps = Vec::new();
    for b in banks {
        for e in 0..experts {
            let gu = Leg::of(params, &b.gate_up, &b.gate_up_scales, e, experts)?;
            let dn = Leg::of(params, &b.down, &b.down_scales, e, experts)?;
            ps.push((gu.top() - 127, dn.top() - 127 - 2));
        }
    }
    Ok(ps)
}

fn experts<T: Word>(x: &Experts, params: &dyn Params, a: &mut [T]) -> R<()> {
    let (py, mx, nex) = (x.rows, x.cols, x.experts);
    let (hc, ic) = (x.hidden / py, x.inter / mx);
    let (xg, xd) = ((hc / 2) * 2 * ic, (ic / 2) * hc);
    let xe = xg + xd;
    let per_pe = x.banks.len() * nex * xe;
    if a.len() != py * mx * per_pe {
        return Err(format!(
            "experts of {per_pe} words a PE, the array's {}",
            a.len() / (py * mx).max(1)
        ));
    }
    let ps = exponents(params, &x.banks, nex)?;
    let jobs: Vec<(usize, usize)> = (0..x.banks.len())
        .flat_map(|l| (0..nex).map(move |e| (l, e)))
        .collect();
    for batch in jobs.chunks(THREADS) {
        let packed: Vec<R<Vec<u16>>> = std::thread::scope(|s| {
            let hs: Vec<_> = batch
                .iter()
                .map(|&(l, e)| {
                    let (b, (pg, pd)) = (&x.banks[l], ps[l * nex + e]);
                    s.spawn(move || -> R<Vec<u16>> {
                        let gu = Leg::of(params, &b.gate_up, &b.gate_up_scales, e, nex)?;
                        let dn = Leg::of(params, &b.down, &b.down_scales, e, nex)?;
                        let what = format!("{} expert {e}", b.gate_up);
                        let bgu = gu.e5m2(pg, &what)?;
                        let bd = dn.e5m2(pd, &format!("{} expert {e}", b.down))?;
                        let (h, i) = (x.hidden, x.inter);
                        // gate row j dim d: bgu[j h + d]; up: bgu[(i + j) h + d]; down dim d row j: bd[d i + j]
                        let mut out = vec![0u16; py * mx * xe];
                        for r in 0..py {
                            for c in 0..mx {
                                let o = (r * mx + c) * xe;
                                let byte = |di: usize, j2: usize| -> u16 {
                                    let d = r * hc + di;
                                    let j = if j2 < ic {
                                        c * ic + j2
                                    } else {
                                        i + c * ic + j2 - ic
                                    };
                                    u16::from(bgu[j * h + d])
                                };
                                for i2 in 0..hc / 2 {
                                    for j2 in 0..2 * ic {
                                        out[o + i2 * 2 * ic + j2] =
                                            (byte(i2, j2) << 8) | byte(i2 + hc / 2, j2);
                                    }
                                }
                                for j2 in 0..ic / 2 {
                                    for di in 0..hc {
                                        let d = r * hc + di;
                                        let (j, jl) = (c * ic + j2, c * ic + j2 + ic / 2);
                                        out[o + xg + j2 * hc + di] = (u16::from(bd[d * i + j])
                                            << 8)
                                            | u16::from(bd[d * i + jl]);
                                    }
                                }
                            }
                        }
                        Ok(out)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("an expert thread panicked".into()))
                })
                .collect()
        });
        for (&(l, e), p) in batch.iter().zip(packed) {
            let p = p?;
            let o0 = (l * nex + e) * xe;
            for pe in 0..py * mx {
                for k in 0..xe {
                    a[pe * per_pe + o0 + k] = T::raw(p[pe * xe + k]);
                }
            }
        }
    }
    Ok(())
}

fn expert_scales<T: Word>(elem: Elem, x: &ExpertScales, params: &dyn Params, a: &mut [T]) -> R<()> {
    let ps = exponents(params, &x.banks, x.experts)?;
    let per = ps.len() * 2;
    if a.len() != x.cols * per {
        return Err(format!("{} rows of {per}, the array's {}", x.cols, a.len()));
    }
    for c in 0..x.cols {
        for (k, (pg, pd)) in ps.iter().enumerate() {
            a[c * per + 2 * k] = T::of(elem, exp2i(*pg));
            a[c * per + 2 * k + 1] = T::of(elem, exp2i(*pd));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{half_to_f32, narrow};

    #[test]
    fn halves_read_back() {
        for (h, v) in [
            (0x3c00u16, 1.0f32),
            (0xc000, -2.0),
            (0x0001, 5.9604645e-08),
            (0x7bff, 65504.0),
        ] {
            assert_eq!(half_to_f32(h), v);
        }
    }

    /// f64 -> f16 through `narrow` rounds once: just past a half step goes up (a plain f32 cast
    /// lands on the tie and rounds to even, down).
    #[test]
    fn narrowing_rounds_once() {
        let x = 1.0 + 2f64.powi(-11) + 2f64.powi(-40);
        assert_eq!(crate::tiles::f16(x as f32), 0x3c00);
        assert_eq!(crate::tiles::f16(narrow(x)), 0x3c01);
        assert_eq!(narrow(1.5), 1.5);
    }
}
