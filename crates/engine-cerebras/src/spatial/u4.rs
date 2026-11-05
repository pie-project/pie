//! u4g64 weights on the wafer: the host half of `csl/kernels/u4mv.csl`.
//!
//! A deployment in `U4g64` (affine 4-bit codes, one bf16 scale and bias per
//! 64 consecutive input dims of a row) keeps its projections 4-bit in PE
//! memory; the unit kernels decode them inside the matvec. The codes are
//! Pie's own: [`Q4::quantize`] is the checkpoint executor's `Encode`
//! (`checkpoint::executor::walk`, MLX's affine parameters), so a bf16
//! checkpoint served as `qwen3-8b-u4g64-kv-bf16` gets the codes, scales and
//! biases every other Pie engine reads for that deployment.
//!
//! A PE's tile of one matrix is `R` rows (one per x element it holds) by `N`
//! outputs. In the weight symbol, after the leading pad word (u16 words):
//!
//! ```text
//! [ng, nseg] [Tile::segs x (woff, len, r0, plane << 8 | group)]  the segments
//! per layer: codes [P][N + 1]                       P = ceil(R / 4)
//!            per group slot: scales [N], biases [N] (bf16 bits)
//! ```
//!
//! Word `(p, j)` holds rows `p, P + p, 2P + p, 3P + p` at bits 0, 4, 8, 12;
//! column `N` holds code 1 for every row of a dim (the group's sum of x). A segment
//! is a run of at most [`Tile::chunk`] rows in one plane and one group, in row
//! order, so the groups come in order and the kernel keeps one accumulator.
//! The kernel computes `y += Σ_g s_g (Σ_{r∈g} x_r q_r) + b_g Σ_{r∈g} x_r`,
//! MLX's `qmv`. A PE's bytes past its codes and scales are a constant: the
//! decode buffer ([`BUF_WORDS`]), one accumulator, x's parts.

use super::ckpt::Ckpt;

type R<T> = std::result::Result<T, String>;

/// Input dims a scale and bias cover.
pub const GROUP: usize = 64;

/// The decode buffer (u16 words; `u4mv.csl`'s `W4BUFW`): a segment's rows
/// decode into it, at most `BUF_WORDS / (N + 1)` of them.
pub const BUF_WORDS: usize = 1024;

/// One matrix quantized: `rows` outputs by `cols` inputs, a code (0..16)
/// per element, a bf16 scale and bias per row and group of 64 inputs.
#[derive(Debug, Clone)]
pub struct Q4 {
    pub rows: usize,
    pub cols: usize,
    codes: Vec<u8>,
    scales: Vec<u16>,
    biases: Vec<u16>,
}

/// f32 to bf16 bits, nearest even (finite values).
fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

fn bf16_round(v: f32) -> f64 {
    f64::from(bf16_f32(bf16_bits(v)))
}

fn bf16_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

impl Q4 {
    /// `w` (`rows` x `cols` bf16 bits, row-major) quantized as the
    /// checkpoint executor's `Encode` does: each group's MLX affine scale and
    /// bias in f32, codes `round((w - bias) / scale)` clamped to 0..15, the
    /// scale and bias stored as bf16.
    pub fn quantize(w: &[u16], rows: usize, cols: usize) -> R<Q4> {
        if w.len() != rows * cols || !cols.is_multiple_of(GROUP) {
            return Err(format!(
                "u4g64: a {rows} x {cols} matrix ({} elements) does not split into groups of {GROUP}",
                w.len()
            ));
        }
        let groups = cols / GROUP;
        let mut q = Q4 {
            rows,
            cols,
            codes: vec![0; rows * cols],
            scales: vec![0; rows * groups],
            biases: vec![0; rows * groups],
        };
        let mut vals = [0f64; GROUP];
        for (gi, chunk) in w.chunks(GROUP).enumerate() {
            for (v, &b) in vals.iter_mut().zip(chunk) {
                *v = f64::from(bf16_f32(b));
            }
            let (scale, bias) = checkpoint::codec::mlx::mlx_affine_group_params_bits(&vals, 4);
            for (k, &v) in vals.iter().enumerate() {
                q.codes[gi * GROUP + k] =
                    (((v as f32) - bias) / scale).round().clamp(0.0, 15.0) as u8;
            }
            q.scales[gi] = bf16_bits(scale);
            q.biases[gi] = bf16_bits(bias);
        }
        Ok(q)
    }

    /// Param `name` of `ckpt` in u4g64 (its shape checked): as landed
    /// ([`Ckpt::u4g64`]), else its bf16 weight quantized.
    pub fn of(ckpt: &Ckpt, name: &str, rows: usize, cols: usize) -> R<Q4> {
        if let Some(l) = ckpt.u4g64(name)? {
            if (l.rows, l.cols) != (rows, cols) || !cols.is_multiple_of(GROUP) {
                return Err(format!(
                    "{name}: landed u4g64 {} x {}, the program wants {rows} x {cols}",
                    l.rows, l.cols
                ));
            }
            let codes = l.codes.iter().flat_map(|&b| [b & 15, b >> 4]).collect();
            return Ok(Q4 {
                rows,
                cols,
                codes,
                scales: l.scales,
                biases: l.biases,
            });
        }
        let bits = super::serve8b::matrix(ckpt, name, rows, cols)?;
        Q4::quantize(&bits, rows, cols)
    }

    #[must_use]
    pub fn code(&self, o: usize, i: usize) -> u8 {
        self.codes[o * self.cols + i]
    }

    /// Output `o`'s scale and bias (bf16 bits) for group `g`.
    #[must_use]
    pub fn affine(&self, o: usize, g: usize) -> (u16, u16) {
        let k = o * (self.cols / GROUP) + g;
        (self.scales[k], self.biases[k])
    }

    /// The dequantized weight, `scale · code + bias` in f32.
    #[must_use]
    pub fn value(&self, o: usize, i: usize) -> f32 {
        let (s, b) = self.affine(o, i / GROUP);
        bf16_f32(s) * f32::from(self.code(o, i)) + bf16_f32(b)
    }
}

/// One matrix's tile shape on a unit PE: `r` rows, `n` outputs, at most `g`
/// groups (`u4mv.csl`'s `W4Rv`, `W4Nv`, `W4Gv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub r: usize,
    pub n: usize,
    pub g: usize,
}

impl Tile {
    #[must_use]
    pub fn planes(&self) -> usize {
        self.r.div_ceil(4)
    }
    /// Plane rows a segment decodes at most (`u4mv.csl`'s `w4k`): the
    /// decode buffer holds [`BUF_WORDS`].
    #[must_use]
    pub fn chunk(&self) -> usize {
        (BUF_WORDS / (self.n + 1)).clamp(1, self.planes().max(1))
    }
    /// Segment slots (`w4segs`): the planes' chunks, split again where a
    /// group starts.
    #[must_use]
    pub fn segs(&self) -> usize {
        (4 * self.planes().div_ceil(self.chunk()) + self.g).saturating_sub(1)
    }
    #[must_use]
    pub fn header(&self) -> usize {
        2 + 4 * self.segs()
    }
    #[must_use]
    pub fn layer_words(&self) -> usize {
        self.planes() * (self.n + 1) + 2 * self.g * self.n
    }
    /// The symbol's u16 words past its pad, for `nl` layers (none for an
    /// absent tile: wqv and wv when fused into wqa).
    #[must_use]
    pub fn words(&self, nl: usize) -> usize {
        if self.r == 0 {
            return 0;
        }
        self.header() + nl * self.layer_words()
    }
}

/// Groups a run of `n` consecutive input dims starting at a multiple of
/// `n` can touch (`u4mv.csl`'s `w4runs`).
fn run_groups(n: usize) -> usize {
    if GROUP.is_multiple_of(n) {
        1
    } else if n.is_multiple_of(GROUP) {
        n / GROUP
    } else {
        (n + 62) / 64 + 1
    }
}

/// The six tiles of a unit PE (`wqa wqv wv wo wgu wd`), as `u4mv.csl`
/// sizes them: `hc` hidden dims a row, `dc` kv dims a column (`gs`
/// columns a head), `qh` query heads a kv head, `ic` intermediate dims.
/// `fused` (B 1, `W4F`): wqa is `[wqa | wqv | wv]`, wqv and wv are absent.
#[must_use]
pub fn tiles(hc: usize, dc: usize, gs: usize, qh: usize, ic: usize, fused: bool) -> [Tile; 6] {
    let dh = dc / 2;
    let hd = gs * dc;
    let gh = run_groups(hc);
    let go = if GROUP.is_multiple_of(dh) && (hd / 2).is_multiple_of(GROUP) {
        2 * qh
    } else {
        2 * qh * run_groups(dh)
    };
    let qk = Tile {
        r: hc,
        n: (qh + 1) * dh,
        g: gh,
    };
    let none = Tile { r: 0, n: 0, g: 0 };
    let (qa, qv, v) = if fused {
        (
            Tile {
                n: 2 * (qh + 1) * dh + dc,
                ..qk
            },
            none,
            none,
        )
    } else {
        (
            qk,
            qk,
            Tile {
                r: hc,
                n: dc,
                g: gh,
            },
        )
    };
    [
        qa,
        qv,
        v,
        Tile {
            r: qh * dc,
            n: hc,
            g: go,
        },
        Tile {
            r: hc,
            n: 2 * ic,
            g: gh,
        },
        Tile {
            r: ic,
            n: hc,
            g: run_groups(ic),
        },
    ]
}

/// Bytes a unit PE's u4 weights and u4 scratch take for `nl` layers (the
/// planner's byte model; the bf16 tiles' `2 · elements` otherwise).
#[must_use]
pub fn pe_bytes(
    hc: usize,
    dc: usize,
    gs: usize,
    qh: usize,
    ic: usize,
    nl: usize,
    fused: bool,
) -> usize {
    let t = tiles(hc, dc, gs, qh, ic, fused);
    let symbols: usize = t.iter().map(|t| 2 * (t.words(nl) + 1)).sum();
    let nmax = t.iter().map(|t| t.n).max().unwrap_or(0);
    let buf = t.iter().map(|t| t.chunk() * (t.n + 1)).max().unwrap_or(0);
    let xmax = hc.max(qh * dc).max(ic);
    // The decode buffer, one accumulator, a group's widened scales, x's
    // parts for planes 1 and 2 (NP 1).
    symbols + 2 * (buf + 1) + 4 * (nmax + 1) + 4 * nmax + 2 * 2 * xmax
}

/// A unit PE's scale and bias bytes a layer, by tile: as the kernel holds
/// them (every group the PE's rows touch: `Tile::g`), and owned (each
/// (output, group) pair on one of the PEs sharing it; a run of `r` rows
/// can be given at most `ceil(r / 64)` groups, every PE of the axis owning
/// some). The difference is what a cross-PE per-group reduction saves.
#[must_use]
pub fn scale_bytes(t: &[Tile; 6]) -> [(usize, usize); 6] {
    t.map(|t| {
        if t.r == 0 {
            (0, 0)
        } else {
            (4 * t.g * t.n, 4 * t.n * t.g.min(t.r.div_ceil(GROUP)))
        }
    })
}

/// A tile's column: output row `o` of a quantized matrix, or padding.
pub type Col<'a> = Option<(&'a Q4, usize)>;

/// One PE's symbol words past the pad (u16 each, one to a u32) for tile
/// `t`: `rows` its input dims (`None`: zero padding), `cols[l]` layer `l`'s
/// outputs. Every layer's rows are the same dims, so one header serves all.
pub fn blob(t: Tile, rows: &[Option<usize>], cols: &[Vec<Col>], out: &mut Vec<u32>) -> R<()> {
    if rows.len() != t.r || cols.iter().any(|c| c.len() != t.n) {
        return Err(format!(
            "u4 tile {t:?}: {} rows, columns {:?}",
            rows.len(),
            cols.iter().map(Vec::len).collect::<Vec<_>>()
        ));
    }
    // Group slots in row order; a padding row joins the slot before it.
    let mut groups: Vec<usize> = Vec::new();
    let mut slot = vec![0usize; t.r];
    for (r, d) in rows.iter().enumerate() {
        if let Some(d) = d {
            let g = d / GROUP;
            if groups.last() != Some(&g) {
                if groups.contains(&g) {
                    return Err(format!("u4 tile {t:?}: group {g} is not one run of rows"));
                }
                groups.push(g);
            }
        }
        slot[r] = groups.len().saturating_sub(1);
    }
    if groups.is_empty() {
        groups.push(0);
    }
    if groups.len() > t.g {
        return Err(format!(
            "u4 tile {t:?}: the rows touch {} groups",
            groups.len()
        ));
    }
    let (p, kr) = (t.planes(), t.chunk());
    let mut segs: Vec<[usize; 3]> = Vec::new();
    for (r, &sl) in slot.iter().enumerate().take(t.r) {
        match segs.last_mut() {
            Some(s) if s[2] == sl && !(r % p).is_multiple_of(kr) => s[1] += 1,
            _ => segs.push([r, 1, sl]),
        }
    }
    if segs.len() > t.segs() {
        return Err(format!("u4 tile {t:?}: {} segments", segs.len()));
    }
    let start = out.len();
    out.push(groups.len() as u32);
    out.push(segs.len() as u32);
    // Segment records: the codes' word offset in a layer, the rows, the
    // first row, plane << 8 | group slot.
    for i in 0..t.segs() {
        out.extend(segs.get(i).map_or([0; 4], |&[r0, len, g]| {
            let k = r0 / p;
            [
                ((r0 - k * p) * (t.n + 1)) as u32,
                len as u32,
                r0 as u32,
                (k << 8 | g) as u32,
            ]
        }));
    }
    for layer in cols {
        let code = |r: usize, c: usize| -> u32 {
            if r >= t.r {
                return 0;
            }
            if c == t.n {
                return u32::from(rows[r].is_some());
            }
            match (rows[r], layer[c]) {
                (Some(i), Some((q, o))) => u32::from(q.code(o, i)),
                _ => 0,
            }
        };
        for pp in 0..p {
            for c in 0..=t.n {
                out.push(
                    code(pp, c)
                        | code(p + pp, c) << 4
                        | code(2 * p + pp, c) << 8
                        | code(3 * p + pp, c) << 12,
                );
            }
        }
        for gs in 0..t.g {
            for half in 0..2 {
                for col in layer {
                    let v = match (groups.get(gs), col) {
                        (Some(&g), Some((q, o))) if gs < groups.len() => {
                            let (s, b) = q.affine(*o, g);
                            if half == 0 { s } else { b }
                        }
                        _ => 0,
                    };
                    out.push(u32::from(v));
                }
            }
        }
    }
    debug_assert_eq!(out.len() - start, t.words(cols.len()));
    Ok(())
}

/// The host's reading of one PE's blob back: `y = A x` for layer `l`, the
/// kernel's arithmetic in f64 (the tests' reference).
#[must_use]
pub fn apply(t: Tile, words: &[u32], l: usize, x: &[f32]) -> Vec<f64> {
    let w = words;
    let (ng, nseg) = (w[0] as usize, w[1] as usize);
    let lb = t.header() + l * t.layer_words();
    let p = t.planes();
    let mut acc = vec![vec![0f64; t.n + 1]; ng];
    for s in 0..nseg {
        let (woff, len, r0, pg) = (
            w[2 + 4 * s] as usize,
            w[3 + 4 * s] as usize,
            w[4 + 4 * s] as usize,
            w[5 + 4 * s] as usize,
        );
        let (k, g) = (pg >> 8, pg & 0xff);
        debug_assert_eq!(woff, (r0 - k * p) * (t.n + 1));
        for r in r0..r0 + len {
            let xr = bf16_round(x[r]);
            for c in 0..=t.n {
                let q = (w[lb + (r - k * p) * (t.n + 1) + c] >> (4 * k)) & 15;
                acc[g][c] += xr * f64::from(q);
            }
        }
    }
    let mut y = vec![0f64; t.n];
    for (g, a) in acc.iter().enumerate() {
        let so = lb + p * (t.n + 1) + 2 * g * t.n;
        for j in 0..t.n {
            let s = f64::from(bf16_f32(w[so + j] as u16));
            let b = f64::from(bf16_f32(w[so + t.n + j] as u16));
            y[j] += s * a[j] + b * a[t.n];
        }
    }
    y
}

/// A PE's tile rows (input dims) and its layers' columns, by (py, px).
type Rows<'q> = dyn Fn(usize, usize) -> (Vec<Option<usize>>, Vec<Vec<Col<'q>>>) + 'q;

#[allow(clippy::too_many_arguments)]
fn put_tile(
    k: super::serve8b::Kernel,
    dev: &mut dyn super::device::Device,
    sym: &str,
    at: (usize, usize),
    nl: usize,
    tile: Tile,
    f: &Rows<'_>,
) -> R<()> {
    let err = std::cell::RefCell::new(None);
    super::serve8b::put_pe16(k, dev, sym, at, tile.words(nl), |py, px, out| {
        let (rows, cols) = f(py, px);
        let start = out.len();
        if let Err(e) = blob(tile, &rows, &cols, out) {
            // Keep the symbol's length; the error is returned after.
            err.borrow_mut()
                .get_or_insert(format!("u4g64 placement: {sym} at ({px}, {py}): {e}"));
            out.truncate(start);
            out.resize(start + tile.words(nl), 0);
        }
    })?;
    err.into_inner().map_or(Ok(()), Err)
}

/// A unit's six weight symbols in u4g64 (`layers` stacked, one header):
/// the same rows and columns as [`super::serve8b::unit`]'s bf16 tiles
/// (column px: kv head px / Gs's dims `dl`, its Qh query heads; row py:
/// hidden dims [py Hc, (py+1) Hc), padding past HR), each tile one
/// [`blob`].
pub(crate) fn place_unit(
    lay: &super::program::ProgramSpec,
    ckpt: &Ckpt,
    k: super::serve8b::Kernel,
    layers: &[usize],
    at: (usize, usize),
    dev: &mut dyn super::device::Device,
) -> R<()> {
    let (pxn, hc, dc, gs, ic) = (k.px, k.hc, k.dc, k.gs, k.ic);
    let (qh, hid, nl) = (lay.qh(), lay.hidden(), layers.len());
    let (d, dh, nh) = (gs * dc, dc / 2, qh + 1);
    let nkv = pxn / gs;
    let (nq, inter) = (nkv * qh, pxn * ic);
    let fused = lay.paged().is_some_and(|s| s.lanes == 1);
    let t = tiles(hc, dc, gs, qh, ic, fused);
    let name = |l: usize, w: &str| format!("layers.{l}.{w}");
    let mats = |w: &str, rows: usize, cols: usize| -> R<Vec<Q4>> {
        layers
            .iter()
            .map(|&l| Q4::of(ckpt, &name(l, w), rows, cols))
            .collect()
    };
    let dl = |px: usize| -> Vec<usize> {
        let j = px % gs;
        (j * dh..(j + 1) * dh)
            .chain((j * dh..(j + 1) * dh).map(|i| d / 2 + i))
            .collect()
    };
    let dim = |py: usize, kk: usize| Some(py * hc + kk).filter(|&x| x < hid);
    let hidden_rows = |py: usize| -> Vec<Option<usize>> { (0..hc).map(|kk| dim(py, kk)).collect() };
    let put = |dev: &mut dyn super::device::Device, sym: &str, tile: Tile, f: &Rows<'_>| {
        put_tile(k, dev, sym, at, nl, tile, f)
    };
    let wq = mats("q", nq * d, hid)?;
    let wk = mats("k", nkv * d, hid)?;
    let wv = mats("v", nkv * d, hid)?;
    // Layer li's columns of wqa (half 0) or wqv (half dh): the q heads' and
    // k's dims of that RoPE half; of wv: v's dims.
    let qk_cols = |li: usize, px: usize, half: usize| -> Vec<Col<'_>> {
        let (g, dl) = (px / gs, dl(px));
        (0..nh)
            .flat_map(|h| (0..dh).map(move |i| (h, i)))
            .map(|(h, i)| {
                let dd = dl[half + i];
                Some(if h < qh {
                    (&wq[li], (g * qh + h) * d + dd)
                } else {
                    (&wk[li], g * d + dd)
                })
            })
            .collect()
    };
    let v_cols = |li: usize, px: usize| -> Vec<Col<'_>> {
        let g = px / gs;
        dl(px)
            .iter()
            .map(|&dd| Some((&wv[li], g * d + dd)))
            .collect()
    };
    if fused {
        put(dev, "wqa", t[0], &|py, px| {
            let cols = (0..nl)
                .map(|li| {
                    let mut c = qk_cols(li, px, 0);
                    c.extend(qk_cols(li, px, dh));
                    c.extend(v_cols(li, px));
                    c
                })
                .collect();
            (hidden_rows(py), cols)
        })?;
    } else {
        for (m, half) in [(0usize, 0usize), (1, dh)] {
            let sym = ["wqa", "wqv"][m];
            put(dev, sym, t[m], &|py, px| {
                let cols = (0..nl).map(|li| qk_cols(li, px, half)).collect();
                (hidden_rows(py), cols)
            })?;
        }
        put(dev, "wv", t[2], &|py, px| {
            let cols = (0..nl).map(|li| v_cols(li, px)).collect();
            (hidden_rows(py), cols)
        })?;
    }
    drop((wq, wk, wv));
    let wo = mats("o", hid, nq * d)?;
    put(dev, "wo", t[3], &|py, px| {
        let (g, dl) = (px / gs, dl(px));
        let rows = (0..qh)
            .flat_map(|h| dl.iter().map(move |&dd| Some((g * qh + h) * d + dd)))
            .collect();
        let cols = wo
            .iter()
            .map(|w| (0..hc).map(|kk| dim(py, kk).map(|o| (w, o))).collect())
            .collect();
        (rows, cols)
    })?;
    drop(wo);
    let wg = mats("gate", inter, hid)?;
    let wu = mats("up", inter, hid)?;
    put(dev, "wgu", t[4], &|py, px| {
        let cols = (0..nl)
            .map(|li| {
                [&wg[li], &wu[li]]
                    .into_iter()
                    .flat_map(|w| (0..ic).map(move |i| Some((w, px * ic + i))))
                    .collect()
            })
            .collect();
        (hidden_rows(py), cols)
    })?;
    drop((wg, wu));
    let wd = mats("down", hid, inter)?;
    put(dev, "wd", t[5], &|py, px| {
        let rows = (0..ic).map(|i| Some(px * ic + i)).collect();
        let cols = wd
            .iter()
            .map(|w| (0..hc).map(|kk| dim(py, kk).map(|o| (w, o))).collect())
            .collect();
        (rows, cols)
    })?;
    drop(wd);
    Ok(())
}

/// `name` (a `rows` x `cols` matrix, `rows` read from the checkpoint) in
/// u4g64: as landed, else its bf16 weight quantized.
fn q4_whole(ckpt: &Ckpt, name: &str, cols: usize) -> R<Q4> {
    if let Some(l) = ckpt.u4g64(name)? {
        return Q4::of(ckpt, name, l.rows, cols);
    }
    let (shape, bits) = ckpt.bits(name)?;
    if shape.len() != 2 || shape[1] != cols {
        return Err(format!(
            "{name} is {shape:?}; the program wants [_, {cols}]"
        ));
    }
    Q4::quantize(&bits, shape[0], cols)
}

/// The head's u4 tile at `W4H` 1 (`codegen::w4_head`): a head PE's `hh2`
/// dims of x (a run from a multiple of `hh2`) by its `vr` vocab rows.
#[must_use]
pub fn head_tile(hh2: usize, vr: usize) -> Tile {
    Tile {
        r: hh2,
        n: vr,
        g: run_groups(hh2),
    }
}

/// u32 words of a head PE's `wt` at `W4H` 1 (`W4TW`): a pad and the
/// blob, u16 words two to a u32 (element 2k in the low half).
#[must_use]
pub fn head_words(t: Tile) -> usize {
    (t.words(1) + 2) / 2
}

/// u16 words `w` (a pad first) packed two to a u32 into `n32` words.
fn pack16(w: &[u32], n32: usize, out: &mut Vec<u32>) {
    for k in 0..n32 {
        let at = |i: usize| w.get(i).copied().unwrap_or(0) & 0xffff;
        out.push(at(2 * k) | at(2 * k + 1) << 16);
    }
}

/// The head's lm_head in u4g64 (`W4H` 1): head PE (chunk column c, region
/// row ry) holds dims `c hh2 ..` of vocab rows `ry Vr ..` as one [`blob`]
/// ([`head_tile`]) after a pad word in `wt`; the VM column zero.
pub(crate) fn place_head(
    lay: &super::program::ProgramSpec,
    ckpt: &Ckpt,
    dev: &mut dyn super::device::Device,
) -> R<()> {
    let (vr, hid, hh2) = (lay.vr(), lay.hidden(), 2 * lay.hc());
    let q = q4_whole(ckpt, "head", hid)?;
    if q.rows > lay.vocab() || lay.vocab() > lay.hh() * vr {
        return Err(format!(
            "head is [{}, {hid}]; the head holds {} rows ({} words)",
            q.rows,
            lay.vocab(),
            lay.hh() * vr
        ));
    }
    let t = head_tile(hh2, vr);
    let n32 = head_words(t);
    let cols = super::serve8b::head_cols(lay);
    let mut err = None;
    let mut tile = |row: usize, c: usize, out: &mut Vec<u32>| {
        let rows: Vec<Option<usize>> = (0..hh2)
            .map(|k| Some(c * hh2 + k).filter(|&d| d < hid))
            .collect();
        let col: Vec<Col> = (0..vr)
            .map(|vv| Some(row * vr + vv).filter(|&o| o < q.rows).map(|o| (&q, o)))
            .collect();
        let mut w = vec![0u32];
        if let Err(e) = blob(t, &rows, &[col], &mut w) {
            err.get_or_insert(format!("u4g64 head at chunk column {c}, row {row}: {e}"));
        }
        pack16(&w, n32, out);
    };
    let words = {
        let mut all = Vec::with_capacity(lay.hh() * cols.len() * n32);
        for row in 0..lay.hh() {
            for c in &cols {
                match c {
                    Some(c) => tile(row, *c, &mut all),
                    None => all.extend(std::iter::repeat_n(0, n32)),
                }
            }
        }
        all
    };
    if let Some(e) = err {
        return Err(e);
    }
    let w = cols.len() * n32;
    super::serve8b::region_rows(lay, dev, "wt", lay.head_x(), cols.len(), n32, |r0, r1| {
        words[r0 * w..r1 * w].to_vec()
    })
}

/// The embedding strip's row layout at `W4H` 1 (`codegen::w4_embed`) for
/// `he` dims a PE: the groups a run of `he` dims from a multiple of `he`
/// touches (`E4G`), a vocab row's u16 words (`E4RW`), the PE's u32 words
/// for `ve` rows.
#[must_use]
pub fn embed_layout(he: usize, ve: usize) -> (usize, usize, usize) {
    let g = run_groups(he);
    let rw = he / 4 + 2 * g;
    (g, rw, (2 + ve * rw).div_ceil(2))
}

/// One strip PE's u16 words at `W4H` 1: `[ph, 0]` (its first dim `d0` mod
/// 64), then for each of its `ve` vocab rows from `v0` the codes of dims
/// `d0 .. d0 + he` (word i: dims i, he/4 + i, ... at bits 0, 4, 8, 12),
/// then the row's scales and biases of the groups from `d0 / 64`.
#[must_use]
pub fn embed_words(q: &Q4, hid: usize, d0: usize, he: usize, v0: usize, ve: usize) -> Vec<u32> {
    let (g, rw, _) = embed_layout(he, ve);
    let hq = he / 4;
    let mut w = Vec::with_capacity(2 + ve * rw);
    w.extend([(d0 % GROUP) as u32, 0]);
    for vv in 0..ve {
        let row = v0 + vv;
        let real = row < q.rows;
        let code = |d: usize| -> u32 {
            if real && d < hid {
                u32::from(q.code(row, d))
            } else {
                0
            }
        };
        for i in 0..hq {
            w.push(
                (0..4)
                    .map(|pl| code(d0 + pl * hq + i) << (4 * pl))
                    .fold(0, |a, b| a | b),
            );
        }
        for half in 0..2 {
            for sl in 0..g {
                let gi = d0 / GROUP + sl;
                let v = if real && gi * GROUP < hid {
                    let (s, b) = q.affine(row, gi);
                    if half == 0 { s } else { b }
                } else {
                    0
                };
                w.push(u32::from(v));
            }
        }
    }
    w
}

/// The kernel's reading of a strip row (`e4_row`): dim k of row `j` as
/// `scale * code + bias` in f32.
#[must_use]
pub fn embed_row(w: &[u32], he: usize, ve: usize, j: usize) -> Vec<f32> {
    let (g, rw, _) = embed_layout(he, ve);
    let (hq, ph) = (he / 4, w[0] as usize);
    let r = &w[2 + j * rw..2 + (j + 1) * rw];
    (0..he)
        .map(|k| {
            let (pl, i) = (k / hq, k % hq);
            let q = (r[i] >> (4 * pl)) & 15;
            let sl = (ph + k) >> 6;
            let s = bf16_f32(r[hq + sl] as u16);
            let b = bf16_f32(r[hq + g + sl] as u16);
            s * q as f32 + b
        })
        .collect()
}

/// The embedding strip in u4g64 (`W4H` 1): strip PE (column EC-1-c, region
/// row ry) holds dims `[c He, (c+1) He)` of vocab rows `ry Ve ..`
/// ([`embed_words`]) in `etab`.
pub(crate) fn place_embed(
    lay: &super::program::ProgramSpec,
    ckpt: &Ckpt,
    dev: &mut dyn super::device::Device,
) -> R<()> {
    let (ve, hid) = (lay.vr(), lay.hidden());
    let (ec, he) = (lay.py() / 4, 4 * lay.hc());
    let q = q4_whole(ckpt, "embed", hid)?;
    if q.rows > lay.hh() * ve {
        return Err(format!(
            "embed is [{}, {hid}]; the strip holds {} rows",
            q.rows,
            lay.hh() * ve
        ));
    }
    let n32 = embed_layout(he, ve).2;
    let xe = lay.head_x() + lay.chunk_cols() + lay.vmc();
    super::serve8b::region_rows(lay, dev, "etab", xe, ec, n32, |r0, r1| {
        let mut t = Vec::with_capacity((r1 - r0) * ec * n32);
        for row in r0..r1 {
            for s in 0..ec {
                let c = ec - 1 - s;
                pack16(&embed_words(&q, hid, c * he, he, row * ve, ve), n32, &mut t);
            }
        }
        t
    })
}

#[cfg(test)]
mod tests {

    /// A W4H build's head and strip, placed through `serve8b::head` /
    /// `serve8b::embed` on the recorder, read back PE by PE: each head PE's
    /// `wt` (the kernel's `W4TW` words) is its dims-by-vocab-rows tile of
    /// the quantized lm_head, each strip PE's `etab` its rows' dequantized
    /// embeddings; padding past the vocab and the hidden size reads 0.
    #[test]
    fn w4h_head_and_strip_read_back() {
        use super::super::codegen::Design;
        use super::super::device::Recorder;
        use super::super::program::ProgramSpec;
        let base = "PX:8,PY:16,Hc:4,Dc:8,Gs:8,Ic:24,RPX:16,RPY:24,RHc:3,RDc:4,RGs:16,RIc:12,RNL:2,\
            RGY:8,RC:2,RR:2,NCOL:2,RTR:5,HR:64,Qh:4,PS:4,NB:2,E:2,B:1,MC:1,W16:1,NP:1,KV16:1,KVCP:0,\
            PROF:0,TIME:0,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:32,Vr:32,Gx:4,Gy:8,vocab:1000,\
            PASSES:3,MODE:0,OVL:0,VMC:1,CAP:16,iters:1,W4:1,W4H:1";
        let kv: Vec<(String, String)> = base
            .split(',')
            .filter_map(|p| p.split_once(':'))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        let lay = ProgramSpec::from_params(Design::Serve8bRegion, &kv).unwrap();
        assert!(lay.head_u4());
        let ckpt = Ckpt::synthetic(super::super::golden::shapes(&lay, true));
        let mut dev = Recorder::default();
        super::super::serve8b::head(&lay, &ckpt, &mut dev).unwrap();
        super::super::serve8b::embed(&lay, &ckpt, &mut dev).unwrap();
        let hid = lay.hidden();
        let (vr, hh2, he) = (lay.vr(), 2 * lay.hc(), 4 * lay.hc());
        let (shape, bits) = ckpt.bits("head").unwrap();
        let qh = Q4::quantize(&bits, shape[0], hid).unwrap();
        let (shape, bits) = ckpt.bits("embed").unwrap();
        let qe = Q4::quantize(&bits, shape[0], hid).unwrap();
        let unpack =
            |w: Vec<u32>| -> Vec<u32> { w.iter().flat_map(|&x| [x & 0xffff, x >> 16]).collect() };
        let (f, cols) = (lay.feeder(), super::super::serve8b::head_cols(&lay));
        let t = head_tile(hh2, vr);
        let (mut heads, mut strips) = (0, 0);
        for ry in 0..lay.hh() {
            let y = if ry < f { ry } else { ry + 1 };
            for (i, c) in cols.iter().enumerate() {
                let Some(c) = *c else { continue };
                let w = dev
                    .memory("wt", (lay.head_x() + i) as i32, y as i32)
                    .unwrap();
                assert_eq!(w.len(), head_words(t));
                let u = unpack(w);
                let x: Vec<f32> = (0..hh2)
                    .map(|k| bf16_f32(bf16_bits(((k + c) as f32 * 0.7).cos())))
                    .collect();
                let got = apply(t, &u[1..], 0, &x);
                for (vv, g) in got.iter().enumerate() {
                    let o = ry * vr + vv;
                    let want: f64 = (0..hh2)
                        .filter(|&k| o < qh.rows && c * hh2 + k < hid)
                        .map(|k| f64::from(qh.value(o, c * hh2 + k)) * f64::from(x[k]))
                        .sum();
                    assert!(
                        (g - want).abs() < 1e-4,
                        "head ({c}, {ry}) row {o}: {g} vs {want}"
                    );
                }
                heads += 1;
            }
            let ec = lay.py() / 4;
            let xe = lay.head_x() + lay.chunk_cols() + lay.vmc();
            for s in 0..ec {
                let c = ec - 1 - s;
                let w = dev.memory("etab", (xe + s) as i32, y as i32).unwrap();
                assert_eq!(w.len(), embed_layout(he, vr).2);
                let u = unpack(w);
                for j in 0..vr {
                    let o = ry * vr + j;
                    for (k, g) in embed_row(&u, he, vr, j).iter().enumerate() {
                        let d = c * he + k;
                        let want = if o < qe.rows && d < hid {
                            qe.value(o, d)
                        } else {
                            0.0
                        };
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "strip ({c}, {ry}) row {o} dim {d}"
                        );
                    }
                }
                strips += 1;
            }
        }
        assert!(heads > 0 && strips > 0);
    }

    /// The strip's u4 rows read back (`embed_row`, the kernel's `e4_row`)
    /// equal the dequantized weight bit for bit, at every group phase and
    /// past the hidden size and the vocab.
    #[test]
    fn embed_rows_read_back_exactly() {
        let (rows, hid) = (7, 320);
        let w: Vec<u16> = (0..rows * hid)
            .map(|i| bf16_bits(((i * 37 % 101) as f32 - 50.0) * 0.013))
            .collect();
        let q = Q4::quantize(&w, rows, hid).unwrap();
        for he in [16, 64, 144, 128] {
            for c in 0..hid.div_ceil(he) {
                let (d0, ve) = (c * he, 4);
                for v0 in [0, 4] {
                    let words = embed_words(&q, hid, d0, he, v0, ve);
                    assert_eq!(words.len(), 2 + ve * embed_layout(he, ve).1);
                    for j in 0..ve {
                        let got = embed_row(&words, he, ve, j);
                        for (k, g) in got.iter().enumerate() {
                            let (o, d) = (v0 + j, d0 + k);
                            let want = if o < rows && d < hid {
                                q.value(o, d)
                            } else {
                                0.0
                            };
                            assert_eq!(
                                g.to_bits(),
                                want.to_bits(),
                                "he {he} c {c} row {o} dim {d}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The head's tile at its shapes: a blob of `head_words` u32 words,
    /// read back (`apply`) as the dense dequantized matvec.
    #[test]
    fn head_tile_reads_back() {
        let (vocab, hid) = (50, 192);
        let w: Vec<u16> = (0..vocab * hid)
            .map(|i| bf16_bits(((i * 53 % 97) as f32 - 48.0) * 0.02))
            .collect();
        let q = Q4::quantize(&w, vocab, hid).unwrap();
        for (hh2, vr) in [(72, 12), (40, 9), (64, 16)] {
            let t = head_tile(hh2, vr);
            for c in 0..hid.div_ceil(hh2) {
                for row in 0..vocab.div_ceil(vr) {
                    let rows: Vec<Option<usize>> = (0..hh2)
                        .map(|k| Some(c * hh2 + k).filter(|&d| d < hid))
                        .collect();
                    let col: Vec<Col> = (0..vr)
                        .map(|vv| Some(row * vr + vv).filter(|&o| o < vocab).map(|o| (&q, o)))
                        .collect();
                    let mut b = Vec::new();
                    blob(t, &rows, &[col], &mut b).unwrap();
                    assert_eq!((b.len() + 2) / 2, head_words(t));
                    let x: Vec<f32> = (0..hh2)
                        .map(|k| bf16_f32(bf16_bits((k as f32 * 0.37).sin())))
                        .collect();
                    let y = apply(t, &b, 0, &x);
                    for (vv, yv) in y.iter().enumerate() {
                        let o = row * vr + vv;
                        let want: f64 = (0..hh2)
                            .filter(|&k| o < vocab && c * hh2 + k < hid)
                            .map(|k| f64::from(q.value(o, c * hh2 + k)) * f64::from(x[k]))
                            .sum();
                        assert!(
                            (yv - want).abs() < 1e-4,
                            "hh2 {hh2} c {c} row {o}: {yv} vs {want}"
                        );
                    }
                }
            }
        }
    }
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
    }

    /// The tile's arithmetic equals the dequantized matrix's product over
    /// the PE's rows (MLX's affine form, f64 here).
    #[test]
    fn blob_applies_the_dequantized_matrix() {
        let (rows, cols) = (40, 256);
        let mut seed = 7;
        let w: Vec<u16> = (0..rows * cols)
            .map(|_| bf16_bits(lcg(&mut seed)))
            .collect();
        let q = Q4::quantize(&w, rows, cols).unwrap();
        // Quantization error within a step (MLX's scale rounds the range's
        // edge onto a code; the far end can clamp).
        for o in 0..rows {
            for i in 0..cols {
                let (s, _) = q.affine(o, i / GROUP);
                let err = (q.value(o, i) - bf16_f32(w[o * cols + i])).abs();
                assert!(err <= bf16_f32(s).abs() * 1.01 + 1e-6, "{o} {i} {err}");
            }
        }
        // A PE holding inputs 50..79 (groups 0 and 1, one padding row) and
        // outputs 3, 5, ... of two layers.
        let t = Tile { r: 30, n: 7, g: 2 };
        let dims: Vec<Option<usize>> = (50..80).map(|d| (d < 79).then_some(d)).collect();
        let cols1: Vec<Col> = (0..7).map(|j| Some((&q, 3 + 2 * j))).collect();
        let cols2: Vec<Col> = (0..7).map(|j| Some((&q, 20 + j))).collect();
        let mut out = Vec::new();
        blob(t, &dims, &[cols1, cols2], &mut out).unwrap();
        assert_eq!(out.len(), t.words(2));
        let x: Vec<f32> = (0..30).map(|i| (i as f32 * 0.37).sin()).collect();
        for (l, base) in [(0usize, 3usize), (1, 20)] {
            let y = apply(t, &out, l, &x);
            for (j, &yj) in y.iter().enumerate().take(7) {
                let o = if l == 0 { base + 2 * j } else { base + j };
                let want: f64 = (0..29)
                    .map(|r| f64::from(q.value(o, 50 + r)) * bf16_round(x[r]))
                    .sum();
                assert!((yj - want).abs() < 1e-6 * want.abs().max(1.0), "{l} {j}");
            }
        }
    }

    /// A matrix landed in u4g64 (`<base>.codes` / `.scales` / `.biases` in a
    /// safetensors file) reads back as the quantizer's own codes.
    #[test]
    fn landed_u4g64_reads_as_quantized() {
        let (rows, cols) = (6, 128);
        let mut seed = 11;
        let w: Vec<u16> = (0..rows * cols)
            .map(|_| bf16_bits(lcg(&mut seed)))
            .collect();
        let q = Q4::quantize(&w, rows, cols).unwrap();
        let packed: Vec<u8> = q.codes.chunks(2).map(|c| c[0] | c[1] << 4).collect();
        let le = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let base = "model.layers.0.mlp.down_proj";
        let parts: Vec<(String, &str, Vec<usize>, Vec<u8>)> = vec![
            (format!("{base}.codes"), "U8", vec![rows, cols / 2], packed),
            (
                format!("{base}.scales"),
                "BF16",
                vec![rows, cols / 64],
                le(&q.scales),
            ),
            (
                format!("{base}.biases"),
                "BF16",
                vec![rows, cols / 64],
                le(&q.biases),
            ),
        ];
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, dtype, shape, bytes) in &parts {
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
            );
            data.extend_from_slice(bytes);
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(h);
        file.extend(data);
        let dir = std::env::temp_dir().join(format!("u4landed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        let ckpt = Ckpt::open(&dir).unwrap();
        let r = Q4::of(&ckpt, "layers.0.down", rows, cols).unwrap();
        assert_eq!((r.codes, r.scales, r.biases), (q.codes, q.scales, q.biases));
        assert!(Q4::of(&ckpt, "layers.0.down", rows, 2 * cols).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Qwen3-8B's grid and region tiles; the O rows' groups.
    #[test]
    fn tiles_of_the_8b_units() {
        let g = tiles(29, 8, 16, 4, 96, false);
        assert_eq!(g[3], Tile { r: 32, n: 29, g: 8 });
        assert_eq!((g[0].g, g[5].g), (2, 3));
        let r = tiles(18, 4, 32, 4, 48, false);
        assert_eq!(r[3], Tile { r: 16, n: 18, g: 8 });
        // u4 vs bf16 bytes a grid PE (weights only).
        let bf16 = 2 * 29 * ((2 * 4 + 2) * 8 + 3 * 96);
        let u4 = pe_bytes(29, 8, 16, 4, 96, 1, false);
        // Fused at B 1: one tile of all three, the same codes.
        let f = tiles(29, 8, 16, 4, 96, true);
        assert_eq!((f[0].n, f[1].words(1), f[2].words(1)), (48, 0, 0));
        assert!(u4 < bf16, "{u4} {bf16}");
    }

    /// serve8b_region's placement in u4 (W4 1) holds, PE by PE, every
    /// entry of its bf16 placement's tiles (within a quantization step):
    /// the same rows and columns in every unit, layer and matrix; at B 1
    /// wqa's tile is the bf16 wqa, wqv and wv tiles side by side.
    #[test]
    fn u4_units_hold_the_bf16_tiles() {
        use super::super::codegen::Design;
        use super::super::device::{Device, Recorder};
        use super::super::program::{Layers, ProgramSpec};
        use super::super::wafer::rect;
        for b in [2, 1] {
            let base = format!(
                "PX:8,PY:16,Hc:4,Dc:8,Gs:8,Ic:24,RPX:16,RPY:24,RHc:3,RDc:4,RGs:16,RIc:12,RNL:2,\
                RGY:8,RC:2,RR:2,NCOL:2,RTR:5,HR:64,Qh:4,PS:4,NB:2,E:2,B:{b},MC:1,W16:1,NP:1,KV16:1,KVCP:0,\
                PROF:0,TIME:0,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:32,Vr:32,Gx:4,Gy:8,vocab:1000,\
                PASSES:3,MODE:0,OVL:0,VMC:1,CAP:16,iters:1"
            );
            let kv = |s: &str| -> Vec<(String, String)> {
                s.split(',')
                    .filter_map(|p| p.split_once(':'))
                    .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                    .collect()
            };
            let bf = ProgramSpec::from_params(Design::Serve8bRegion, &kv(&base)).unwrap();
            let w4 = ProgramSpec::from_params(Design::Serve8bRegion, &kv(&format!("{base},W4:1")))
                .unwrap();
            assert!(w4.weights_u4() && !bf.weights_u4());
            let Layers::Units { units, .. } = &bf.layers else {
                panic!("units")
            };
            let ckpt = Ckpt::synthetic(super::super::golden::shapes(&bf, true));
            let rope = super::super::wafer::Rotary::theta(1.0e6);
            let (mut a, mut q) = (Recorder::default(), Recorder::default());
            super::super::serve8b_region::place_layers(&bf, units, &ckpt, &rope, &mut a).unwrap();
            super::super::serve8b_region::place_layers(&w4, units, &ckpt, &rope, &mut q).unwrap();
            let fused = b == 1;
            let mut checked = 0;
            for v in units.units(&bf) {
                let k = v.k;
                let nl = v.layers.len();
                let t = tiles(k.hc, k.dc, k.gs, bf.qh(), k.ic, false);
                let tq = tiles(k.hc, k.dc, k.gs, bf.qh(), k.ic, fused);
                let r = rect(v.x0, v.y0, k.px, k.py);
                let syms = ["wqa", "wqv", "wv", "wo", "wgu", "wd"];
                // The u4 symbol holding bf16 matrix m, and its first column there.
                let home = |m: usize| -> (usize, usize) {
                    match m {
                        1 if fused => (0, t[0].n),
                        2 if fused => (0, t[0].n + t[1].n),
                        _ => (m, 0),
                    }
                };
                for (m, sym) in syms.iter().enumerate() {
                    let (hm, c0) = home(m);
                    let elems = t[m].r * t[m].n;
                    let qw = tq[hm].words(nl) + 1;
                    let dense = a.get16(sym, r, nl * elems + 1).unwrap();
                    let quant = q.get16(syms[hm], r, qw).unwrap();
                    let span = dense
                        .iter()
                        .map(|&w| bf16_f32(w as u16))
                        .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
                    let tol = f64::from(span.1 - span.0) / 15.0 * 1.05 + 1e-6;
                    for pe in 0..k.px * k.py {
                        let d = &dense[pe * (nl * elems + 1) + 1..(pe + 1) * (nl * elems + 1)];
                        let qb = &quant[pe * qw + 1..(pe + 1) * qw];
                        for l in 0..nl {
                            for row in 0..t[m].r {
                                let mut x = vec![0f32; t[m].r];
                                x[row] = 1.0;
                                let y = apply(tq[hm], qb, l, &x);
                                for c in 0..t[m].n {
                                    let want =
                                        f64::from(bf16_f32(d[l * elems + row * t[m].n + c] as u16));
                                    assert!(
                                        (y[c0 + c] - want).abs() <= tol,
                                        "B {b} {sym} unit {} pe {pe} layer {l} ({row}, {c}): {} vs {want}",
                                        v.idx,
                                        y[c0 + c]
                                    );
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
            assert!(checked > 100_000, "{checked}");
        }
    }

    /// `U4_DUMP=<ckpt dir>:<param>:<out file>`: the param quantized, written
    /// as codes (u8 a code), then scales and biases (bf16 bits, LE), to
    /// check another quantizer (the golden's) against this one bit for bit.
    #[test]
    #[ignore]
    fn dump_quantized() {
        let spec = std::env::var("U4_DUMP").expect("U4_DUMP");
        let mut it = spec.split(':');
        let (dir, param, out) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        let ckpt = Ckpt::open(std::path::Path::new(dir)).unwrap();
        let (shape, bits) = ckpt.bits(param).unwrap();
        let q = Q4::quantize(&bits, shape[0], shape[1]).unwrap();
        let mut bytes = q.codes.clone();
        for v in q.scales.iter().chain(&q.biases) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(out, bytes).unwrap();
    }
}
