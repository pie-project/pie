//! What the host writes into a stream program's PEs before a step and reads after it: every
//! [`HostArray`]'s words, from a step's values (the weights, the host's rows and tables: what
//! engine-cpu's `Forward::stream_capture` holds, `n<node>_in<k>.npy` / `_out<k>.npy` with
//! `nodes.json`), laid out as the kernels read them, grouped into memcpy rectangles
//! (`host.json`, read by `tools/wafer/stream/run.py`).
//!
//! The P11 permutations are the host's: a root's rows are its own rows in the order the row
//! stream brings them (its schedule's), each the class's row of the step; the entry's context is
//! clamped to f16's range in an IEEE build.
//!
//! The reading's entries (M1e): an array whose words a step's inputs make (the host's rows, each
//! row's scalars: what the entry's host ops make of the latents, the context, the timestep) is
//! also stated as a [`Gather`], each word a value's (row, column), so an engine that computes
//! those values from its own inputs ([`apply`]) writes the array as the capture does, a step at
//! a time. `host.json` names the reading's classes (their request words and rows), the entries
//! and the values the host reads back, so a runner needs nothing else of the compile.

use std::collections::BTreeMap;
use std::path::Path;

use poem_ir::ValueId;
use serde_json::json;

use super::build::Program;
use super::unit::{Cols, Side, Step};
use crate::emit::stream::{Fill, HostArray};
use crate::floor::Coord;

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

    pub fn get(&self, v: ValueId) -> R<Held> {
        if let Some(x) = self.cache.borrow().get(&v.0) {
            return Ok(x.clone());
        }
        let f = self
            .files
            .get(&v.0)
            .ok_or_else(|| format!("value %{} is not captured", v.0))?;
        let (s, d) = crate::npy::read_f32_or_i32(&self.dir.join(f))?;
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
fn half_bits(x: f32) -> u16 {
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

/// The trace row of a unit's own row `t`.
fn trace_row(p: &Program, unit: usize, t: u32) -> usize {
    match &p.units[unit].sched {
        Some(s) => {
            let (c, k) = s.rows[t as usize];
            (p.class_rows[c].1 + k) as usize
        }
        None => 0,
    }
}

/// One array's words (f32, or 16-bit words in u32).
fn fill(p: &Program, vals: &Values, a: &HostArray) -> R<Vec<f32>> {
    let mut out = vec![0f32; a.len as usize];
    let ieee = !p.round16;
    match &a.fill {
        Fill::Tile {
            unit,
            step,
            col,
            row,
        } => {
            let b = &p.units[*unit];
            let l = &b.layout;
            let Step::Matvec {
                from,
                input,
                weight,
                output,
                k,
                n,
                ..
            } = &b.unit.steps[*step]
            else {
                return Err("a tile of no matvec".into());
            };
            let (_, w) = vals.get(*weight)?;
            let kk = *k as usize;
            let (rows, cols): (Vec<usize>, Vec<usize>) = match from {
                Side::Row => {
                    let a = k / l.py;
                    let outs = l.parts(
                        *col,
                        Cols {
                            base: *output,
                            from: 0,
                            width: *n,
                        },
                    );
                    (
                        (row * a..(row + 1) * a).map(|x| x as usize).collect(),
                        outs.iter()
                            .flat_map(|(_, s)| (s.from..s.from + s.len).map(|x| x as usize))
                            .collect(),
                    )
                }
                Side::Col => {
                    let b2 = n / l.py;
                    let ins = l.parts(*col, *input);
                    (
                        ins.iter()
                            .flat_map(|(_, s)| {
                                (s.from - input.from..s.from - input.from + s.len)
                                    .map(|x| x as usize)
                            })
                            .collect(),
                        (row * b2..(row + 1) * b2).map(|x| x as usize).collect(),
                    )
                }
            };
            // tile[r][c] = W[out c, in r]
            for (ri, &xr) in rows.iter().enumerate() {
                for (ci, &oc) in cols.iter().enumerate() {
                    let wv = w[oc * kk + xr];
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
        Fill::Resident { .. } | Fill::Scalars { .. } | Fill::HostIn { .. } => {
            let g = gather(p, a)?.ok_or("an array of no gather")?;
            out = apply(&g, &|v| {
                let (s, d) = vals.get(ValueId(v))?;
                let w = *s.last().unwrap_or(&1);
                Ok((d.len() / w.max(1), w, d))
            })?;
        }
        Fill::HostOut { .. } | Fill::Stamps => {}
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

/// `g`'s words from the values `get` holds: (rows, width, elements) by value id.
pub fn apply(
    g: &[Word],
    get: &dyn Fn(u32) -> R<(usize, usize, std::rc::Rc<Vec<f32>>)>,
) -> R<Vec<f32>> {
    let mut held: BTreeMap<u32, (usize, usize, std::rc::Rc<Vec<f32>>)> = BTreeMap::new();
    let mut out = Vec::with_capacity(g.len());
    for &[v, row, col, flags] in g {
        if v == NONE {
            out.push(0.0);
            continue;
        }
        if !held.contains_key(&v) {
            held.insert(v, get(v)?);
        }
        let (rows, width, d) = &held[&v];
        let at = if flags & FLAT != 0 {
            col as usize % d.len().max(1)
        } else {
            let r = if flags & ONE != 0 && *rows == 1 {
                0
            } else {
                row as usize
            };
            r * width + col as usize
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
                    for e in 0..len {
                        g[(t * width + o + e) as usize] = [base, row, from + e, one | clamp];
                    }
                    o += len;
                }
            }
        }
        _ => return Ok(None),
    }
    Ok(Some(g))
}

/// Whether `v` is made from the step's inputs (so the host writes it each step), not from the
/// weights alone.
fn of_step(trace: &poem_ir::Trace, v: ValueId, memo: &mut BTreeMap<u32, bool>) -> bool {
    use poem_ir::{Def, Operands};
    if let Some(&m) = memo.get(&v.0) {
        return m;
    }
    memo.insert(v.0, false);
    let r = match &trace.values[v.0 as usize].def {
        Def::Input(_) => true,
        Def::Weight(_) | Def::Cache(_) => false,
        Def::Merge(arms) => arms.iter().any(|&(a, _)| of_step(trace, a, memo)),
        Def::Op(n) => {
            let mut ins = Vec::new();
            trace.nodes[*n as usize].op.inputs(&mut ins);
            ins.into_iter().any(|i| of_step(trace, i, memo))
        }
    };
    memo.insert(v.0, r);
    r
}

/// The arrays grouped into memcpy rectangles: each group one symbol and length, its PEs a
/// rectangle (rows of consecutive PEs, stacked where they repeat).
fn rects(arrays: &[&HostArray]) -> Vec<(u32, u32, u32, u32, Vec<usize>)> {
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

/// Writes the host's files for `p`'s `arrays` into `out` from the step's values in `gold`:
/// `h2d/<k>.npy` and `host.json`.
pub fn write(p: &Program, arrays: &[HostArray], gold: &Path, out: &Path) -> R<()> {
    let vals = Values::open(gold)?;
    std::fs::create_dir_all(out.join("h2d")).map_err(|e| e.to_string())?;
    let ins: Vec<&HostArray> = arrays
        .iter()
        .filter(|a| !matches!(a.fill, Fill::HostOut { .. } | Fill::Stamps))
        .collect();
    std::fs::create_dir_all(out.join("entries")).map_err(|e| e.to_string())?;
    let mut memo = BTreeMap::new();
    let mut h2d = Vec::new();
    for (k, (x, y, w, h, members)) in rects(&ins).into_iter().enumerate() {
        let a0 = ins[members[0]];
        let mut data: Vec<f32> = Vec::with_capacity(members.len() * a0.len as usize);
        for &m in &members {
            data.extend(fill(p, &vals, ins[m])?);
        }
        let file = format!("h2d/{k}.npy");
        if a0.u16 {
            let words: Vec<u32> = data.iter().map(|x| x.to_bits()).collect();
            crate::npy::write_u32(
                &out.join(&file),
                &[h as usize, w as usize, a0.len as usize],
                &words,
            )?;
        } else {
            crate::npy::write_f32(
                &out.join(&file),
                &[h as usize, w as usize, a0.len as usize],
                &data,
            )?;
        }
        let mut item = json!({"sym": a0.sym, "x": x, "y": y, "w": w, "h": h, "len": a0.len, "u16": a0.u16, "file": file});
        // an entry: its words gathered from the step's values, a step at a time
        let mut words: Vec<u32> = Vec::new();
        let mut step = false;
        for &m in &members {
            match gather(p, ins[m])? {
                Some(g) => {
                    step |= g
                        .iter()
                        .any(|w| w[0] != NONE && of_step(&p.trace, ValueId(w[0]), &mut memo));
                    words.extend(g.iter().flatten());
                }
                None => words.clear(),
            }
        }
        if step && words.len() == members.len() * a0.len as usize * 4 {
            let gfile = format!("entries/{k}.npy");
            crate::npy::write_u32(
                &out.join(&gfile),
                &[h as usize, w as usize, a0.len as usize, 4],
                &words,
            )?;
            item["entry"] = json!(gfile);
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
    let ref_files: Vec<serde_json::Value> = refs
        .iter()
        .map(|&b| json!({"value": b, "file": vals.files.get(&b)}))
        .collect();
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
    let host = json!({
        "model": p.trace.name, "reading": p.shape.reading, "classes": classes, "outs": outs,
        "w": p.stream.w, "h": p.stream.h, "h2d": h2d, "d2h": d2h, "refs": ref_files,
        "gold": gold.display().to_string(), "round16": p.round16,
        "units": p.units.iter().zip(&p.stream.units).map(|(b, su)| json!({
            "name": b.unit.name, "x": su.x, "cols": su.cols(), "rows": p.stream.py,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        out.join("host.json"),
        serde_json::to_string_pretty(&host).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
