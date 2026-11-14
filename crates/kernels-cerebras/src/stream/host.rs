//! What the host writes into a stream program's PEs before a step and reads after it: every
//! [`HostArray`]'s words, from a step's values (the weights, the host's rows and tables: what
//! engine-cpu's `Forward::stream_capture` holds, `n<node>_in<k>.npy` / `_out<k>.npy` with
//! `nodes.json`), laid out as the kernels read them, grouped into memcpy rectangles
//! (`host.json`, read by `tools/wafer/stream/run.py`).
//!
//! The P11 permutations are the host's: a root's rows are its own rows in the order the row
//! stream brings them (its schedule's), each the class's row of the step; the entry's context is
//! clamped to f16's range in an IEEE build.

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
    let trace = &p.trace;
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
            let (_, v) = vals.get(input.value)?;
            let wlen = v.len();
            for e in 0..at.chunk {
                let col = (input.from + r * at.chunk + e) as usize;
                out[e as usize] = v[col % wlen];
            }
        }
        Fill::Scalars { unit, of } => {
            let b = &p.units[*unit];
            let ns = of.len();
            for t in 0..b.unit.rows {
                let tr = trace_row(p, *unit, t);
                for (i, &(step, param)) in of.iter().enumerate() {
                    let Step::Region { chain, .. } = &b.unit.steps[step] else {
                        return Err("a scalar of no region".into());
                    };
                    let input = &chain.ins[param];
                    let (s, v) = vals.get(input.value)?;
                    let w = *s.last().unwrap_or(&1);
                    let rows = v.len() / w.max(1);
                    let row = if rows == 1 { 0 } else { tr };
                    out[t as usize * ns + i] = v[row * w + input.from as usize];
                }
            }
        }
        Fill::HostIn { unit, parts } => {
            let b = &p.units[*unit];
            let width: u32 = parts.iter().map(|x| x.2).sum();
            for t in 0..b.unit.rows {
                let tr = trace_row(p, *unit, t);
                let mut o = 0;
                for &(base, from, len) in parts {
                    let (s, v) = vals.get(ValueId(base))?;
                    let w = *s.last().unwrap_or(&1);
                    let rows = v.len() / w.max(1);
                    let row = if rows == 1 || b.unit.classes.is_empty() {
                        0
                    } else {
                        tr
                    };
                    for e in 0..len {
                        let mut x = v[row * w + (from + e) as usize];
                        if ieee {
                            x = x.clamp(-65504.0, 65504.0);
                        }
                        out[(t * width + o + e) as usize] = x;
                    }
                    o += len;
                }
            }
        }
        Fill::HostOut { .. } | Fill::Stamps => {}
    }
    let _ = trace;
    Ok(out)
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
        h2d.push(json!({"sym": a0.sym, "x": x, "y": y, "w": w, "h": h, "len": a0.len, "u16": a0.u16, "file": file}));
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
    let host = json!({
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
