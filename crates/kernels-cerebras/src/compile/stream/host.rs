//! What the host writes into a stream program's PEs before a step and reads after it: every
//! [`HostArray`]'s words, laid out as the kernels read them and grouped into memcpy rectangles
//! (`host.json`). Arrays a step's inputs fill are stated as a [`Gather`] (each word a value's
//! row and column, see [`apply`]); arrays the weights fill as [`Span`]s over the trace's values
//! (see [`apply_spans`]), so a program compiled with no capture runs on any checkpoint.

use std::collections::BTreeMap;
use std::path::Path;

use poem_ir::ValueId;
use serde_json::json;

use super::build::Program;
use super::unit::{Cols, Side, Step};
use crate::compile::emit::stream::{Fill, HostArray};
use crate::compile::floor::Coord;

type R<T> = Result<T, String>;

/// A captured value: its shape and values.
pub type Held = (Vec<usize>, std::rc::Rc<Vec<f32>>);

/// A step's captured values by value id: (shape, values).
pub struct Values {
    files: BTreeMap<u32, String>,
    dir: std::path::PathBuf,
    cache: std::cell::RefCell<BTreeMap<u32, Held>>,
}

impl Values {
    pub fn open(dir: &Path) -> R<Values> {
        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("nodes.json"))
                .map_err(|e| format!("nodes.json: {e}"))?,
        )
        .map_err(|e| format!("nodes.json: {e}"))?;
        let mut files = BTreeMap::new();
        for node in index.as_object().ok_or("nodes.json: no object")?.values() {
            for (f, v) in node["values"].as_object().ok_or("nodes.json: no values")? {
                files.insert(v.as_u64().unwrap_or(0) as u32, f.clone());
            }
        }
        Ok(Values {
            files,
            dir: dir.to_path_buf(),
            cache: std::cell::RefCell::new(BTreeMap::new()),
        })
    }

    /// The capture's file holding `v`, if it has one.
    #[must_use]
    pub fn file(&self, v: ValueId) -> Option<&str> {
        self.files.get(&v.0).map(String::as_str)
    }

    pub fn get(&self, v: ValueId) -> R<Held> {
        if let Some(x) = self.cache.borrow().get(&v.0) {
            return Ok(x.clone());
        }
        let f = self
            .files
            .get(&v.0)
            .ok_or_else(|| format!("value %{} is not captured", v.0))?;
        let (s, d) = crate::compile::npy::read_f32_or_i32(&self.dir.join(f))?;
        let x = (s, std::rc::Rc::new(d));
        self.cache.borrow_mut().insert(v.0, x.clone());
        Ok(x)
    }
}

fn bits16(x: f32, ieee: bool) -> u32 {
    if ieee {
        u32::from(half_bits(x))
    } else {
        let b = x.to_bits();
        (b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16
    }
}

/// IEEE f16 bits of `x`, rounded to nearest even.
pub(crate) fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let r = if rem > half || (rem == half && r & 1 == 1) {
            r + 1
        } else {
            r
        };
        return sign | r as u16;
    }
    let r = man >> 13;
    let rem = man & 0x1fff;
    let mut h = ((e as u32) << 10) | r;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// The f32 of IEEE f16 bits `h`.
pub(crate) fn half_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The trace row of a unit's own row `t`.
fn trace_row(p: &Program, unit: usize, t: u32) -> usize {
    match &p.units[unit].sched {
        // (a row its schedule does not list: its own row t)
        Some(s) => s
            .rows
            .get(t as usize)
            .map_or(t as usize, |&(c, k)| (p.class_rows[c].1 + k) as usize),
        None => 0,
    }
}

/// A padded row's or column of a tile ([`tile_cells`]): no weight's, a zero.
const PAD: usize = usize::MAX;

/// A tile's cells: its rows (the weight's input columns), its columns (the weight's output
/// rows; [`PAD`] past a value the row roots pad), the weight's width and the weight.
fn tile_cells(p: &Program, unit: usize, step: usize, col: u32, row: u32) -> R<Cells> {
    let b = &p.units[unit];
    let l = &b.layout;
    let Step::Matvec {
        from,
        input,
        weight,
        view,
        output,
        k,
        n,
        ..
    } = &b.unit.steps[step]
    else {
        return Err("a tile of no matvec".into());
    };
    let (rows, cols): (Vec<usize>, Vec<usize>) = match from {
        Side::Row => {
            // (a value narrower than the row roots, or of no multiple of them, is padded: a
            // padded input's row of the tile is [`PAD`])
            let a = k.div_ceil(l.py);
            let outs = l.parts(
                col,
                Cols {
                    base: *output,
                    from: 0,
                    width: *n,
                },
            );
            (
                (row * a..(row + 1) * a)
                    .map(|x| if x < *k { x as usize } else { PAD })
                    .collect(),
                outs.iter()
                    .flat_map(|(_, s)| (s.from..s.from + s.len).map(|x| x as usize))
                    .collect(),
            )
        }
        Side::Col => {
            let b2 = n.div_ceil(l.py);
            let ins = l.parts(col, *input);
            (
                ins.iter()
                    .flat_map(|(_, s)| {
                        (s.from - input.from..s.from - input.from + s.len).map(|x| x as usize)
                    })
                    .collect(),
                (row * b2..(row + 1) * b2)
                    .map(|x| if x < *n { x as usize } else { PAD })
                    .collect(),
            )
        }
    };
    Ok(Cells {
        rows,
        cols,
        k: *k as usize,
        n: *n as usize,
        weight: *weight,
        view: *view,
    })
}

/// A tile's cells ([`tile_cells`]).
struct Cells {
    rows: Vec<usize>,
    cols: Vec<usize>,
    k: usize,
    n: usize,
    weight: ValueId,
    view: super::unit::View,
}

impl Cells {
    /// Whether the cell of input `xr`, output `oc` is a constant one (an identity's, a spread's).
    fn one(&self, xr: usize, oc: usize) -> bool {
        if xr == PAD || oc == PAD || self.weight.0 != UNIT {
            return false;
        }
        match self.view {
            super::unit::View::Spread { rep } => oc / rep as usize == xr,
            super::unit::View::Sum { rep } => xr / rep as usize == oc,
            _ => xr == oc,
        }
    }

    /// The weight's element the cell is, `None` for a constant or padded one.
    fn at(&self, xr: usize, oc: usize) -> Option<usize> {
        if xr == PAD || oc == PAD || self.weight.0 == UNIT {
            return None;
        }
        self.view.at(oc, xr, self.k, self.n)
    }
}

/// One array's words (f32, or 16-bit words in u32).
fn fill(p: &Program, vals: Option<&Values>, a: &HostArray) -> R<Vec<f32>> {
    let held = || vals.ok_or_else(|| format!("`{}`: a value's array, and no capture", a.sym));
    let mut out = vec![0f32; a.len as usize];
    let ieee = !p.round16;
    match &a.fill {
        Fill::Tile {
            unit,
            step,
            col,
            row,
        } => {
            let cells = tile_cells(p, *unit, *step, *col, *row)?;
            let (rows, cols, weight) = (&cells.rows, &cells.cols, cells.weight);
            if weight.0 == UNIT {
                // an identity's (a spread's) tile: its constant ones
                for (ri, &xr) in rows.iter().enumerate() {
                    for (ci, &oc) in cols.iter().enumerate() {
                        if cells.one(xr, oc) {
                            out[1 + ri * cols.len() + ci] = f32::from_bits(bits16(1.0, ieee));
                        }
                    }
                }
                return Ok(out);
            }
            let (_, w) = held()?.get(weight)?;
            // tile[r][c] = W[out c, in r]
            for (ri, &xr) in rows.iter().enumerate() {
                for (ci, &oc) in cols.iter().enumerate() {
                    let wv = cells.at(xr, oc).map_or(0.0, |e| w[e]);
                    out[1 + ri * cols.len() + ci] = f32::from_bits(bits16(wv, ieee));
                }
            }
        }
        Fill::Table {
            unit,
            step,
            table,
            r,
        } => {
            let t = &p.units[*unit].regions[step].1.tables[*table];
            for (i, x) in t.of_chunk[*r as usize].iter().enumerate() {
                out[i] = if t.u16 { f32::from_bits(*x as u32) } else { *x };
            }
        }
        Fill::Resident { .. }
        | Fill::Scalars { .. }
        | Fill::RowScalars { .. }
        | Fill::Slices(_)
        | Fill::HostIn { .. } => {
            let g = gather(p, a)?.ok_or("an array of no gather")?;
            let vals = held()?;
            out = apply(&g, &|v| {
                let (s, d) = vals.get(ValueId(v))?;
                let w = *s.last().unwrap_or(&1);
                Ok((d.len() / w.max(1), w, d))
            })?;
        }
        Fill::Words(w) => out = w.iter().map(|&x| f32::from_bits(x)).collect(),
        Fill::HostOut { .. } | Fill::HostArgmax { .. } | Fill::Stamps => {}
    }
    Ok(out)
}

/// A word of an entry: `value`'s element at (`row`, `col`), or zero where `value` is [`NONE`];
/// `flags` [`ONE`], [`CLAMP`], [`FLAT`].
pub type Word = [u32; 4];

/// An array's words, each a value's element: what [`apply`] fills it from.
pub type Gather = Vec<Word>;

/// No value: the word is zero.
pub const NONE: u32 = u32::MAX;
/// A value of one row (a lane vector a step: the timestep's embedding) is read at row 0.
pub const ONE: u32 = 1;
/// Clamped to f16's range (an IEEE build's host rows).
pub const CLAMP: u32 = 2;
/// `col` indexes the value's elements over every row, modulo its size (a static resident).
pub const FLAT: u32 = 4;

/// A value as [`apply`] reads it: its rows, width and elements.
pub type Rows = (usize, usize, std::rc::Rc<Vec<f32>>);

/// A run of an array's words: `count` times (the `j`-th from `step · j` elements on), `len`
/// elements of `value` from (`row`, `col`), `stride` apart along the row ([`FLAT`]: `col`
/// indexes the value's elements over every row), each cast as `flags` say ([`ONE`], [`CLAMP`],
/// [`FLAT`], [`HALF`], [`BF16`]); `len · count` zeros where `value` is [`NONE`].
/// `[value, row, col, len, stride, flags, count, step]`.
pub type Span = [u32; 8];

/// No value but the constant one: `len · count` words of 1.0, cast as `flags` say (an identity
/// tile's diagonal).
pub const UNIT: u32 = u32::MAX - 1;

/// No value but a literal: `row` itself, `len · count` times (a 16-bit one with [`PACK`]: a
/// program's words streamed with its weights).
pub const LIT: u32 = u32::MAX - 2;

/// The word is the element's IEEE f16 bits (a tile of an IEEE build).
pub const HALF: u32 = 8;
/// The word is the element's bf16 bits, rounded to nearest even (a tile of a bf16 build).
pub const BF16: u32 = 16;
/// With [`HALF`] or [`BF16`]: two 16-bit elements a word, the first in its low half (a tiled
/// program's weight stream, two weights a wavelet). An array's packed elements pair across its
/// spans; a span that is not packed starts on a whole word.
pub const PACK: u32 = 32;

/// The words `s` makes (its packed spans' elements two a word).
#[must_use]
pub fn span_words(s: &[Span]) -> u32 {
    let mut halves = 0u32;
    let mut words = 0u32;
    for x in s {
        let n = x[3] * x[6];
        if x[5] & PACK != 0 {
            halves += n;
        } else {
            words += halves.div_ceil(2) + n;
            halves = 0;
        }
    }
    words + halves.div_ceil(2)
}

/// `s`'s words (each an f32's bits, or a 16-bit word) from the values `get` holds, by value id.
pub fn apply_spans(s: &[Span], get: &dyn Fn(u32) -> R<Rows>) -> R<Vec<u32>> {
    let mut all = Vec::new();
    apply_spans_into(s, get, usize::MAX, &mut |w| {
        all = w;
        Ok(())
    })?;
    Ok(all)
}

/// [`apply_spans`] a run of `chunk` words at a time, each handed to `emit` as it fills, and each
/// value let go after the last span that reads it: an array's words in bounded memory.
pub fn apply_spans_into(
    s: &[Span],
    get: &dyn Fn(u32) -> R<Rows>,
    chunk: usize,
    emit: &mut dyn FnMut(Vec<u32>) -> R<()>,
) -> R<()> {
    let mut last: BTreeMap<u32, usize> = BTreeMap::new();
    for (i, sp) in s.iter().enumerate() {
        last.insert(sp[0], i);
    }
    let mut held: BTreeMap<u32, Rows> = BTreeMap::new();
    let mut out: Vec<u32> = Vec::new();
    let mut emitted = false;
    // (a run full: handed on)
    let mut push = |out: &mut Vec<u32>, w: u32, emit: &mut dyn FnMut(Vec<u32>) -> R<()>| -> R<()> {
        out.push(w);
        if out.len() == chunk {
            emitted = true;
            emit(std::mem::take(out))?;
        }
        Ok(())
    };
    // a packed element waiting for its word's high half
    let mut half: Option<u32> = None;
    for (si, &[v, row, col, len, stride, flags, count, step]) in s.iter().enumerate() {
        if flags & PACK == 0
            && let Some(lo) = half.take()
        {
            push(&mut out, lo, emit)?;
        }
        if v == NONE {
            for _ in 0..len as usize * count as usize {
                push(&mut out, 0, emit)?;
            }
            continue;
        }
        if v == LIT {
            for _ in 0..len as usize * count as usize {
                if flags & PACK == 0 {
                    push(&mut out, row, emit)?;
                } else if let Some(lo) = half.take() {
                    push(&mut out, lo | ((row & 0xffff) << 16), emit)?;
                } else {
                    half = Some(row & 0xffff);
                }
            }
            continue;
        }
        if v == UNIT {
            let one = if flags & HALF != 0 {
                u32::from(half_bits(1.0))
            } else if flags & BF16 != 0 {
                bits16(1.0, false)
            } else {
                1f32.to_bits()
            };
            for _ in 0..len as usize * count as usize {
                push(&mut out, one, emit)?;
            }
            continue;
        }
        if let std::collections::btree_map::Entry::Vacant(e) = held.entry(v) {
            e.insert(get(v)?);
        }
        let (rows, width, d) = held.get(&v).ok_or("a value held")?.clone();
        let base = if flags & FLAT != 0 {
            col as usize
        } else {
            let r = if flags & ONE != 0 && rows == 1 {
                0
            } else {
                row as usize
            };
            r * width + col as usize
        };
        for i in (0..count as usize).flat_map(|j| (0..len as usize).map(move |i| (j, i))) {
            let mut at = base + i.0 * step as usize + i.1 * stride as usize;
            if flags & FLAT != 0 {
                at %= d.len().max(1);
            }
            let mut x = *d
                .get(at)
                .ok_or_else(|| format!("value %{v}: element {at} of {}", d.len()))?;
            if flags & CLAMP != 0 {
                x = x.clamp(-65504.0, 65504.0);
            }
            let w = if flags & HALF != 0 {
                u32::from(half_bits(x))
            } else if flags & BF16 != 0 {
                bits16(x, false)
            } else {
                x.to_bits()
            };
            if flags & PACK == 0 {
                push(&mut out, w, emit)?;
            } else if let Some(lo) = half.take() {
                push(&mut out, lo | (w << 16), emit)?;
            } else {
                half = Some(w);
            }
        }
        if last.get(&v) == Some(&si) {
            held.remove(&v);
        }
    }
    if let Some(lo) = half {
        out.push(lo);
    }
    if !out.is_empty() || !emitted {
        emit(out)?;
    }
    Ok(())
}

/// `g` as spans: each run of words of one value, row and flags at consecutive columns one.
fn spans_of(g: &[Word]) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    for &[v, row, col, flags] in g {
        if let Some(last) = out.last_mut() {
            let [lv, lr, lc, ln, _, lf, _, _] = *last;
            if lv == v && v == NONE {
                last[3] += 1;
                continue;
            }
            if lv == v && lr == row && lf == flags && lc + ln == col {
                last[3] += 1;
                continue;
            }
        }
        out.push(if v == NONE {
            [NONE, 0, 0, 1, 0, 0, 1, 0]
        } else {
            [v, row, col, 1, 1, flags, 1, 0]
        });
    }
    out
}

/// The words of `a` as runs of values' elements, for an array a value fills (a weight tile, a
/// resident, a row's scalars, a root's host rows); `None` for the rest (a region's table, a PE's
/// config words: the program's own).
pub fn spans(p: &Program, a: &HostArray) -> R<Option<Vec<Span>>> {
    if let Some(g) = gather(p, a)? {
        return Ok(Some(spans_of(&g)));
    }
    let Fill::Tile {
        unit,
        step,
        col,
        row,
    } = &a.fill
    else {
        return Ok(None);
    };
    let cells = tile_cells(p, *unit, *step, *col, *row)?;
    let (rows, cols, kk, weight) = (
        cells.rows.clone(),
        cells.cols.clone(),
        cells.k,
        cells.weight,
    );
    let cast = if p.round16 { BF16 } else { HALF };
    // word 0 is none; then tile[r][c] = W[out c, in r]: each row of the tile a run a stretch
    // of consecutive output columns, the rows one span where they are consecutive inputs and
    // the row is one stretch
    let mut stretches: Vec<(usize, usize)> = Vec::new();
    if rows.contains(&PAD) || cols.contains(&PAD) || cells.view != super::unit::View::Plain {
        // a padded tile, or a view's: each cell a span of its own (a padded one none's, a
        // constant one the unit's)
        let flags = FLAT | cast;
        let mut s: Vec<Span> = vec![[NONE, 0, 0, 1, 0, 0, 1, 0]];
        // a tile row's cells as runs: a weight's elements at one stride, zeros, ones
        for &xr in &rows {
            let mut run: Option<Span> = None;
            for &oc in &cols {
                let cell = match (cells.one(xr, oc), cells.at(xr, oc)) {
                    (true, _) => [UNIT, 0, 0, 1, 0, flags, 1, 0],
                    (false, Some(e)) => [weight.0, 0, e as u32, 1, kk as u32, flags, 1, 0],
                    (false, None) => [NONE, 0, 0, 1, 0, 0, 1, 0],
                };
                match run.as_mut() {
                    // a zero after zeros
                    Some(r) if r[0] == NONE && cell[0] == NONE => r[3] += 1,
                    // a weight's element one stride on from the run's last
                    Some(r)
                        if r[0] == weight.0
                            && cell[0] == weight.0
                            && r[0] != UNIT
                            && (r[3] == 1 || cell[2] == r[2] + r[3] * r[4])
                            && cell[2] > r[2] =>
                    {
                        if r[3] == 1 {
                            r[4] = cell[2] - r[2];
                        }
                        if cell[2] == r[2] + r[3] * r[4] {
                            r[3] += 1;
                        } else {
                            s.push(*r);
                            *r = cell;
                        }
                    }
                    Some(r) => {
                        s.push(*r);
                        *r = cell;
                    }
                    None => run = Some(cell),
                }
            }
            s.extend(run);
        }
        let used = 1 + rows.len() * cols.len();
        if used < a.len as usize {
            s.push([NONE, 0, 0, (a.len as usize - used) as u32, 0, 0, 1, 0]);
        }
        return Ok(Some(s));
    }
    for &c in &cols {
        match stretches.last_mut() {
            Some((from, n)) if *from + *n == c => *n += 1,
            _ => stretches.push((c, 1)),
        }
    }
    let flags = FLAT | cast;
    let mut s: Vec<Span> = vec![[NONE, 0, 0, 1, 0, 0, 1, 0]];
    if weight.0 == UNIT {
        // an identity's tile: one where the input column is the output's, runs of zeros between
        let mut zeros = 0u32;
        for &xr in &rows {
            for &oc in &cols {
                if xr == oc {
                    if zeros > 0 {
                        s.push([NONE, 0, 0, zeros, 0, 0, 1, 0]);
                        zeros = 0;
                    }
                    s.push([UNIT, 0, 0, 1, 0, flags, 1, 0]);
                } else {
                    zeros += 1;
                }
            }
        }
        let used = 1 + rows.len() * cols.len() - zeros as usize;
        let rest = a.len as usize - used;
        if rest > 0 {
            s.push([NONE, 0, 0, rest as u32, 0, 0, 1, 0]);
        }
        return Ok(Some(s));
    }
    let consecutive = rows.windows(2).all(|r| r[1] == r[0] + 1);
    if let ([(c, n)], true, Some(&r0)) = (&stretches[..], consecutive, rows.first()) {
        s.push([
            weight.0,
            0,
            (c * kk + r0) as u32,
            *n as u32,
            kk as u32,
            flags,
            rows.len() as u32,
            1,
        ]);
    } else {
        for &xr in &rows {
            for &(c, n) in &stretches {
                s.push([
                    weight.0,
                    0,
                    (c * kk + xr) as u32,
                    n as u32,
                    kk as u32,
                    flags,
                    1,
                    0,
                ]);
            }
        }
    }
    let used = 1 + rows.len() * cols.len();
    if used < a.len as usize {
        s.push([NONE, 0, 0, (a.len as usize - used) as u32, 0, 0, 1, 0]);
    }
    Ok(Some(s))
}

/// `g`'s words from the values `get` holds, by value id.
pub fn apply(g: &[Word], get: &dyn Fn(u32) -> R<Rows>) -> R<Vec<f32>> {
    let mut held: BTreeMap<u32, Rows> = BTreeMap::new();
    let mut out = Vec::with_capacity(g.len());
    for &[v, row, col, flags] in g {
        if v == NONE {
            out.push(0.0);
            continue;
        }
        let (rows, width, d) = match held.entry(v) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => e.insert(get(v)?),
        };
        let at = if flags & FLAT != 0 {
            col as usize % d.len().max(1)
        } else {
            let r = if flags & ONE != 0 && *rows == 1 {
                0
            } else {
                row as usize
            };
            r * *width + col as usize
        };
        let mut x = *d
            .get(at)
            .ok_or_else(|| format!("value %{v}: element {at} of {}", d.len()))?;
        if flags & CLAMP != 0 {
            x = x.clamp(-65504.0, 65504.0);
        }
        out.push(x);
    }
    Ok(out)
}

/// The words of `a` as values' elements, for an array a step's values fill (a region's
/// resident, a unit's rows' scalars, a root's host rows); `None` for the rest.
pub fn gather(p: &Program, a: &HostArray) -> R<Option<Gather>> {
    let ieee = !p.round16;
    let mut g: Gather = vec![[NONE, 0, 0, 0]; a.len as usize];
    match &a.fill {
        Fill::Slices(parts) => {
            let mut k = 0;
            for &(v, from, len) in parts {
                for e in 0..len {
                    g[k] = [v, 0, from + e, FLAT];
                    k += 1;
                }
            }
        }
        Fill::Resident {
            unit,
            step,
            param,
            r,
        } => {
            let b = &p.units[*unit];
            let Step::Region { chain, .. } = &b.unit.steps[*step] else {
                return Err("a resident of no region".into());
            };
            let (at, _, _) = &b.regions[step];
            let input = &chain.ins[*param];
            for e in 0..at.chunk {
                g[e as usize] = [input.value.0, 0, input.from + r * at.chunk + e, FLAT];
            }
        }
        Fill::Scalars { unit, of } => {
            let b = &p.units[*unit];
            let ns = of.len();
            for t in 0..b.unit.rows {
                let tr = trace_row(p, *unit, t) as u32;
                for (i, &(step, param)) in of.iter().enumerate() {
                    let Step::Region { chain, .. } = &b.unit.steps[step] else {
                        return Err("a scalar of no region".into());
                    };
                    let input = &chain.ins[param];
                    g[t as usize * ns + i] = [input.value.0, tr, input.from, ONE];
                }
            }
        }
        Fill::RowScalars {
            unit,
            of,
            from,
            step,
        } => {
            let b = &p.units[*unit];
            let ns = of.len();
            let mut t = *from;
            let mut k = 0;
            while t < b.unit.rows && (k + 1) * ns <= g.len() {
                let tr = trace_row(p, *unit, t) as u32;
                for (i, &(st, param)) in of.iter().enumerate() {
                    let Step::Region { chain, .. } = &b.unit.steps[st] else {
                        return Err("a scalar of no region".into());
                    };
                    let input = &chain.ins[param];
                    g[k * ns + i] = [input.value.0, tr, input.from, ONE];
                }
                t += step.max(&1);
                k += 1;
            }
        }
        Fill::HostIn { unit, parts } => {
            let b = &p.units[*unit];
            let width: u32 = parts.iter().map(|x| x.2).sum();
            let clamp = if ieee { CLAMP } else { 0 };
            for t in 0..b.unit.rows {
                let tr = trace_row(p, *unit, t) as u32;
                let mut o = 0;
                for &(base, from, len) in parts {
                    // a unit of no class (the lane prologue) reads a lane vector's first row
                    let (row, one) = if b.unit.classes.is_empty() {
                        (0, 0)
                    } else {
                        (tr, ONE)
                    };
                    let whole = width_of_value(&p.trace, base);
                    for e in 0..len {
                        // (past a value the row roots pad: a zero)
                        g[(t * width + o + e) as usize] = if from + e < whole {
                            [base, row, from + e, one | clamp]
                        } else {
                            [NONE, 0, 0, 0]
                        };
                    }
                    o += len;
                }
            }
        }
        _ => return Ok(None),
    }
    Ok(Some(g))
}

/// Whether `v` is made from the step's inputs `hit` picks (any: the host writes it each step),
/// not from the weights alone.
fn of_step(
    trace: &poem_ir::Trace,
    v: ValueId,
    hit: fn(&poem_ir::RuntimeInput) -> bool,
    memo: &mut BTreeMap<u32, bool>,
) -> bool {
    use poem_ir::{Def, Operands};
    if let Some(&m) = memo.get(&v.0) {
        return m;
    }
    memo.insert(v.0, false);
    let r = match &trace.values[v.0 as usize].def {
        Def::Input(i) => hit(i),
        Def::Weight(_) | Def::Cache(_) => false,
        Def::Merge(arms) => arms.iter().any(|&(a, _)| of_step(trace, a, hit, memo)),
        Def::Op(n) => {
            let mut ins = Vec::new();
            trace.nodes[*n as usize].op.inputs(&mut ins);
            ins.into_iter().any(|i| of_step(trace, i, hit, memo))
        }
    };
    memo.insert(v.0, r);
    r
}

/// The arrays grouped into memcpy rectangles: each group one symbol and length, its PEs a
/// rectangle (rows of consecutive PEs, stacked where they repeat).
pub(crate) fn rects(arrays: &[&HostArray]) -> Vec<(u32, u32, u32, u32, Vec<usize>)> {
    let mut by: BTreeMap<(String, u32), Vec<usize>> = BTreeMap::new();
    for (i, a) in arrays.iter().enumerate() {
        by.entry((a.sym.clone(), a.len)).or_default().push(i);
    }
    let mut out = Vec::new();
    for idx in by.values() {
        let at: BTreeMap<Coord, usize> = idx.iter().map(|&i| (arrays[i].at, i)).collect();
        let mut done: std::collections::BTreeSet<Coord> = std::collections::BTreeSet::new();
        let mut cs: Vec<Coord> = at.keys().copied().collect();
        cs.sort_by_key(|c| (c.y, c.x));
        for c in cs {
            if done.contains(&c) {
                continue;
            }
            let mut w = 1;
            while at.contains_key(&Coord::new(c.x + w, c.y))
                && !done.contains(&Coord::new(c.x + w, c.y))
            {
                w += 1;
            }
            let mut h = 1;
            while (0..w).all(|dx| {
                let q = Coord::new(c.x + dx, c.y + h);
                at.contains_key(&q) && !done.contains(&q)
            }) {
                h += 1;
            }
            let mut members = Vec::new();
            for dy in 0..h {
                for dx in 0..w {
                    let q = Coord::new(c.x + dx, c.y + dy);
                    done.insert(q);
                    members.push(at[&q]);
                }
            }
            out.push((c.x, c.y, w, h, members));
        }
    }
    out
}

/// Writes the host's files for `p`'s `arrays` into `out`: `host.json`, each array's spans over
/// the trace's values (`statics/<k>.npy`) or its entry's gather (`entries/<k>.npy`), and the
/// words of `h2d/<k>.npy`: every array's from the step's values in `gold`, else only those no
/// value fills (the program's tables and config words). Given `gold`, each array's spans are
/// held to its words.
pub fn write(p: &Program, arrays: &[HostArray], gold: Option<&Path>, out: &Path) -> R<()> {
    let vals = gold.map(Values::open).transpose()?;
    for d in ["h2d", "entries", "statics"] {
        std::fs::create_dir_all(out.join(d)).map_err(|e| e.to_string())?;
    }
    let ins: Vec<&HostArray> = arrays
        .iter()
        .filter(|a| {
            !matches!(
                a.fill,
                Fill::HostOut { .. } | Fill::HostArgmax { .. } | Fill::Stamps
            )
        })
        .collect();
    let (mut memo, mut other) = (BTreeMap::new(), BTreeMap::new());
    let mut h2d = Vec::new();
    for (k, (x, y, w, h, members)) in rects(&ins).into_iter().enumerate() {
        let a0 = ins[members[0]];
        let shape = [h as usize, w as usize, a0.len as usize];
        let mut item =
            json!({"sym": a0.sym, "x": x, "y": y, "w": w, "h": h, "len": a0.len, "u16": a0.u16});
        // the array's words as runs of values' elements, where a value fills every member
        let mut runs: Option<Vec<Span>> = Some(Vec::new());
        for &m in &members {
            match (spans(p, ins[m])?, runs.as_mut()) {
                (Some(s), Some(all)) => all.extend(s),
                _ => runs = None,
            }
        }
        // an entry: its words gathered from the step's values, a step at a time
        let mut words: Vec<u32> = Vec::new();
        let (mut step, mut early) = (false, true);
        for &m in &members {
            match gather(p, ins[m])? {
                Some(g) => {
                    for w in g.iter().filter(|w| w[0] != NONE) {
                        step |= of_step(&p.trace, ValueId(w[0]), |_| true, &mut memo);
                        let past = |i: &_| !matches!(i, poem_ir::RuntimeInput::Positions);
                        early &= !of_step(&p.trace, ValueId(w[0]), past, &mut other);
                    }
                    words.extend(g.iter().flatten());
                }
                None => words.clear(),
            }
        }
        let entry = step && words.len() == members.len() * a0.len as usize * 4;
        let constant = members
            .iter()
            .all(|&m| matches!(ins[m].fill, Fill::Table { .. } | Fill::Words(_)));
        if vals.is_some() || constant {
            let mut data: Vec<f32> = Vec::with_capacity(members.len() * a0.len as usize);
            for &m in &members {
                data.extend(fill(p, vals.as_ref(), ins[m])?);
            }
            if let (Some(v), Some(r), false) = (&vals, &runs, constant) {
                let got = apply_spans(r, &|id| {
                    let (s, d) = v.get(ValueId(id))?;
                    let w = *s.last().unwrap_or(&1);
                    Ok((d.len() / w.max(1), w, d))
                })?;
                if got.len() != data.len() || got.iter().zip(&data).any(|(g, d)| *g != d.to_bits())
                {
                    return Err(format!(
                        "`{}` at ({x}, {y}): its spans are not its words",
                        a0.sym
                    ));
                }
            }
            let file = format!("h2d/{k}.npy");
            if a0.u16 {
                let words: Vec<u32> = data.iter().map(|x| x.to_bits()).collect();
                crate::compile::npy::write_u32(&out.join(&file), &shape, &words)?;
            } else {
                crate::compile::npy::write_f32(&out.join(&file), &shape, &data)?;
            }
            item["file"] = json!(file);
        }
        if entry {
            let gfile = format!("entries/{k}.npy");
            crate::compile::npy::write_u32(
                &out.join(&gfile),
                &[h as usize, w as usize, a0.len as usize, 4],
                &words,
            )?;
            item["entry"] = json!(gfile);
            // (of the positions alone: the host writes the next step's during this one)
            if early {
                item["early"] = json!(true);
            }
        } else if let (Some(r), false) = (&runs, constant) {
            let sfile = format!("statics/{k}.npy");
            let flat: Vec<u32> = r.iter().flatten().copied().collect();
            crate::compile::npy::write_u32(&out.join(&sfile), &[r.len(), 8], &flat)?;
            item["static"] = json!(sfile);
        } else if !constant && vals.is_none() {
            return Err(format!(
                "`{}` at ({x}, {y}): its words follow from no value and no capture",
                a0.sym
            ));
        }
        h2d.push(item);
    }
    // what is read back: the host rows out (their trace rows and columns), the stamps
    let mut d2h = Vec::new();
    for a in arrays {
        if let Fill::HostOut { unit, parts } = &a.fill {
            let rows: Vec<usize> = (0..p.units[*unit].unit.rows)
                .map(|t| trace_row(p, *unit, t))
                .collect();
            d2h.push(json!({"sym": a.sym, "x": a.at.x, "y": a.at.y, "len": a.len, "rows": rows, "parts": parts}));
        }
    }
    // a decode program's argmax a row, each root's over its columns of the logits
    let mut argmax = Vec::new();
    for a in arrays {
        if let Fill::HostArgmax { unit, parts } = &a.fill {
            let rows: Vec<usize> = (0..p.units[*unit].unit.rows)
                .map(|t| trace_row(p, *unit, t))
                .collect();
            argmax.push(json!({"sym": a.sym, "x": a.at.x, "y": a.at.y, "len": a.len, "rows": rows, "parts": parts}));
        }
    }
    // the reference: the host values the outputs are compared to
    let mut refs = Vec::new();
    for a in arrays {
        if let Fill::HostOut { parts, .. } = &a.fill {
            for &(base, _, _) in parts {
                if !refs.contains(&base) {
                    refs.push(base);
                }
            }
        }
    }
    let ref_files: Vec<serde_json::Value> = match &vals {
        Some(v) => refs
            .iter()
            .map(|&b| json!({"value": b, "file": v.files.get(&b)}))
            .collect(),
        None => Vec::new(),
    };
    // the values read back, whole: the step's rows of each, its columns the parts read
    let step_rows: u32 = p.class_rows.iter().map(|c| c.0).sum();
    let mut outs: BTreeMap<u32, u32> = BTreeMap::new();
    for a in arrays {
        if let Fill::HostOut { parts, .. } = &a.fill {
            for &(base, from, len) in parts {
                let w = outs.entry(base).or_default();
                *w = (*w).max(from + len);
            }
        }
    }
    let outs: Vec<serde_json::Value> = outs
        .iter()
        .map(|(v, w)| json!({"value": v, "rows": step_rows, "width": w}))
        .collect();
    let classes: Vec<serde_json::Value> = p
        .shape
        .program
        .classes
        .iter()
        .map(|c| json!({"name": c.name, "word": c.word.to_string(), "rows": c.rows}))
        .collect();
    let mut host = json!({
        "model": p.trace.name, "reading": p.shape.reading, "classes": classes, "outs": outs,
        "w": p.stream.w, "h": p.stream.h, "h2d": h2d, "d2h": d2h, "refs": ref_files,
        "round16": p.round16, "causal": p.shape.attn.causal,
        "units": p.units.iter().zip(&p.stream.units).map(|(b, su)| json!({
            "name": b.unit.name, "x": su.x, "cols": su.cols(), "rows": p.stream.py,
        })).collect::<Vec<_>>(),
    });
    if !argmax.is_empty() {
        host["argmax"] = json!(argmax);
    }
    if let Some(gold) = gold {
        // (absolute: the runners read it from another directory)
        let gold = std::fs::canonicalize(gold).unwrap_or_else(|_| gold.to_path_buf());
        host["gold"] = json!(gold.display().to_string());
    }
    std::fs::write(
        out.join("host.json"),
        serde_json::to_string_pretty(&host).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// `v`'s row width in `trace` (a moved copy's, its value's).
fn width_of_value(trace: &poem_ir::Trace, v: u32) -> u32 {
    match trace
        .values
        .get((v & !super::unit::MOVED) as usize)
        .map(|d| &d.ty)
    {
        Some(poem_ir::Ty::Tensor { shape, .. }) => match shape.last() {
            Some(poem_ir::Dim::Const(n)) => *n as u32,
            _ => 0,
        },
        _ => 0,
    }
}
