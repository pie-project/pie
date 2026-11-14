//! The stream regime's op-derived step time (GENERIC.md §2.4): each unit's rows priced by what
//! its PEs do a row, the units' phases in the order the row stream and the barriers allow.
//!
//! - P1-stream: the busiest weight PE's tile elements a row x [`P1_STREAM`] (M1c: 0.545 - 0.572
//!   cycles an element, f32 accumulation).
//! - P4 regions: their ops over the chunk at [`REGION_OP`] an element, [`ALL_REDUCE`] a segment
//!   all-reduce, [`ROTATION`] a rotation's P5 digit products (v0's scalar loops, M1c's measured
//!   region cycles).
//! - P7: [`STORE`] a stored row, [`QUERY`] a query (M1c, 16 entries a PE).
//!
//! A unit's phase costs its rows x its slowest PE a row (the PEs of a unit pipeline the rows);
//! the step is the units' phases along the stream: a unit's pre-barrier phase starts when its
//! rows reach it, its post-barrier phase when every unit of its block has stored.

use serde::Serialize;

use crate::stream::build::{Program, kv_pya};
use crate::stream::place;
use crate::stream::unit::{Side, Step};

/// P1-stream cycles an element a row.
pub const P1_STREAM: f64 = 0.55;
/// A region's DSD op: its issue, and an element of it.
pub const DSD_OP: f64 = 20.0;
pub const DSD_ELEM: f64 = 1.0;
/// A vector exp's (math.vexp) and reciprocal's (three Newton steps) elements.
pub const EXP_ELEM: f64 = 12.0;
pub const RECIP_ELEM: f64 = 12.0;
/// A chunk sum's element (the halving tree) and a level of it.
pub const SUM_ELEM: f64 = 1.0;
pub const SUM_LEVEL: f64 = 20.0;
/// A scalar math call (rsqrt, recip, ln, exp of a scalar).
pub const SCALAR_MATH: f64 = 60.0;
/// A segment all-reduce over a unit's CX roots.
pub const ALL_REDUCE: f64 = 1_200.0;
/// A position's cos / sin from P5 tables (one axis: LV levels of NROT-wide products).
pub const ROTATION: f64 = 300.0;
/// P7: a stored row; a query, and an entry a PE keeps.
pub const STORE: f64 = 800.0;
pub const QUERY: f64 = 1_100.0;
pub const QUERY_ENTRY: f64 = 151.0;

/// What a printed region's `row()` does a row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RegionFeatures {
    pub dsd_ops: f64,
    pub dsd_elems: f64,
    pub exp_elems: f64,
    pub recip_elems: f64,
    pub sums: f64,
    pub sum_elems: f64,
    pub scalar_math: f64,
    pub all_reduces: f64,
    pub rotations: f64,
}

impl RegionFeatures {
    /// The features of a printed module `csl` at `chunk` columns a root: each statement of
    /// `row()` (a per-head loop's statements its heads' times, each over a head).
    #[must_use]
    pub fn of(csl: &str, chunk: u32) -> RegionFeatures {
        let mut f = RegionFeatures::default();
        let Some(at) = csl.find("fn row() void {") else {
            return f;
        };
        let c = f64::from(chunk);
        for line in csl[at..].lines().skip(1) {
            if line.starts_with('}') {
                break;
            }
            // a per-head loop: `while (j < H)`
            let reps = line
                .find("while (j < ")
                .and_then(|i| {
                    line[i + 11..]
                        .split(')')
                        .next()
                        .and_then(|n| n.trim().parse::<f64>().ok())
                })
                .unwrap_or(1.0);
            let count = |pat: &str| line.matches(pat).count() as f64;
            let sums = count("vsum(");
            f.sums += sums * reps;
            f.sum_elems += sums * c;
            let exps = count("vexp(");
            f.exp_elems += exps * c;
            let recips = count("vrecip(");
            f.recip_elems += recips * c;
            f.scalar_math += count("math.") * reps - exps - recips;
            f.all_reduces += count("seg.all_reduce(");
            f.rotations += count("rp.tables(");
            let ops = [
                "@fmuls(", "@fadds(", "@fsubs(", "@fmovs(", "@fs2h(", "@fh2s(", "@fnegs(",
                "@fmaxs(", "@fmacs(",
            ]
            .iter()
            .map(|p| count(p))
            .sum::<f64>();
            f.dsd_ops += ops * reps;
            f.dsd_elems += ops * c;
        }
        f.scalar_math = f.scalar_math.max(0.0);
        f
    }

    /// Cycles a row.
    #[must_use]
    pub fn cycles(&self, chunk: u32) -> f64 {
        let levels = f64::from(chunk.max(2)).log2().ceil();
        self.dsd_ops * DSD_OP
            + self.dsd_elems * DSD_ELEM
            + self.exp_elems * EXP_ELEM
            + self.recip_elems * RECIP_ELEM
            + self.sum_elems * SUM_ELEM
            + self.sums * levels * SUM_LEVEL
            + self.scalar_math * SCALAR_MATH
            + self.all_reduces * ALL_REDUCE
            + self.rotations * ROTATION
    }
}

/// One unit's priced phases.
#[derive(Clone, Debug, Serialize)]
pub struct UnitCost {
    pub name: String,
    pub rows: u32,
    /// Cycles a row before / after the barrier (the slowest PE's).
    pub early_row: f64,
    pub late_row: f64,
    /// The weight PEs' share of them.
    pub matvec_row: f64,
}

/// The step's priced units and the step's cycles.
#[derive(Clone, Debug, Serialize)]
pub struct StreamCost {
    pub units: Vec<UnitCost>,
    pub cycles: f64,
}

fn region_cycles(p: &Program, u: usize, step: usize) -> f64 {
    let b = &p.units[u];
    let Some((at, printed, _)) = b.regions.get(&step) else {
        return 0.0;
    };
    RegionFeatures::of(&printed.csl, at.chunk).cycles(at.chunk)
}

/// A query's cycles over a K / V column of `rows` stored rows.
#[must_use]
pub fn query_cycles(rows: u32, q_heads: u32, head_dim: u32) -> f64 {
    let ne = rows.div_ceil(kv_pya(rows, q_heads, head_dim));
    QUERY + QUERY_ENTRY * f64::from(ne * q_heads)
}

/// The step time of `p`.
#[must_use]
pub fn time(p: &Program) -> StreamCost {
    let rows: Vec<u32> = p.class_rows.iter().map(|r| r.0).collect();
    time_at(p, &rows)
}

fn rows_of(classes: &[usize], class_rows: &[u32]) -> u32 {
    classes
        .iter()
        .map(|&c| class_rows.get(c).copied().unwrap_or(0))
        .sum()
}

/// The step time of `p` at `class_rows` rows a class (its own: [`time`]): a unit's rows are its
/// classes', a query's cost its block's entries a PE.
#[must_use]
pub fn time_at(p: &Program, class_rows: &[u32]) -> StreamCost {
    let a = &p.shape.attn;
    let qh = a.q_heads / a.kv_heads.max(1);
    let mut units = Vec::new();
    for (ui, b) in p.units.iter().enumerate() {
        let l = &b.layout;
        let (mut e, mut lt, mut mv) = (0.0f64, 0.0f64, 0.0f64);
        // the weight PEs: the busiest column a row, by phase
        for (phase, acc) in [(false, &mut e), (true, &mut lt)] {
            let busiest = (0..l.px())
                .map(|c| {
                    b.unit
                        .steps
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i >= b.unit.split) == phase)
                        .filter_map(|(_, s)| place::tile_dims(s, l, c))
                        .map(|(a, bb)| f64::from(a * bb) * P1_STREAM)
                        .sum::<f64>()
                })
                .fold(0.0, f64::max);
            mv = mv.max(busiest);
            // the roots: their regions a row, on the slowest side
            let mut row_side = 0.0;
            let mut col_side = 0.0;
            for (i, s) in b.unit.steps.iter().enumerate() {
                if (i >= b.unit.split) != phase {
                    continue;
                }
                match s {
                    Step::Region {
                        side: Side::Row, ..
                    } => row_side += region_cycles(p, ui, i),
                    Step::Region {
                        side: Side::Col, ..
                    } => col_side += region_cycles(p, ui, i),
                    Step::Store { .. } => col_side += STORE,
                    Step::Attend { .. } => {
                        let block: u32 =
                            p.kv.iter().find(|k| k.units.contains(&ui)).map_or(0, |k| {
                                k.units
                                    .iter()
                                    .map(|&v| rows_of(&p.units[v].unit.classes, class_rows))
                                    .sum()
                            });
                        col_side += query_cycles(block, qh, a.head_dim);
                    }
                    _ => {}
                }
            }
            *acc = busiest.max(row_side).max(col_side);
        }
        units.push(UnitCost {
            name: b.unit.name.clone(),
            rows: if b.unit.classes.is_empty() {
                b.unit.rows
            } else {
                rows_of(&b.unit.classes, class_rows)
            },
            early_row: e,
            late_row: lt,
            matvec_row: mv,
        });
    }
    // along the stream: a unit starts its rows once the unit before it has started its own
    // (rows pipeline), its late phase once its block has stored every row
    let mut t = 0.0f64;
    let mut done = 0.0f64;
    for u in &units {
        let early = f64::from(u.rows) * u.early_row;
        let late = f64::from(u.rows) * u.late_row;
        let start = t;
        let end_early = start + early;
        let end = end_early + late;
        done = done.max(end);
        // the next unit's rows arrive as this one's leave: after its barrier where it has one
        t = if late > 0.0 {
            end_early
        } else {
            start + u.early_row
        };
    }
    StreamCost {
        cycles: done,
        units,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A printed row(): a per-head loop's ops count once a head, its elements once a chunk.
    #[test]
    fn a_printed_rows_features() {
        let csl = "fn row() void {\n  @fmuls(B0, IN0, IN0);\n  \
                   { var j: u16 = 0; while (j < 2) : (j += 1) { h7[j] = vsum(&b0, j * 64, 64); } }\n  \
                   { var j: u16 = 0; while (j < 2) : (j += 1) { h12[j] = math.rsqrt(h11[j]); } }\n  \
                   sred[0] = vsum(&b1, 0, C);\n  seg.all_reduce(SRED, 1);\n  \
                   @fs2h(H16, B1); @fh2s(B1, H16);\n}\n";
        let f = RegionFeatures::of(csl, 128);
        assert_eq!((f.dsd_ops, f.dsd_elems), (3.0, 384.0));
        assert_eq!((f.sums, f.sum_elems), (3.0, 256.0));
        assert_eq!((f.scalar_math, f.all_reduces), (2.0, 1.0));
    }

    /// P7 v1's columns: flux at 1024² (4,608 rows, a q head of 128 a kv head) takes about a
    /// hundred attention PEs a kv head, 47 entries each.
    #[test]
    fn klein_s_kv_columns_at_1024() {
        assert_eq!(crate::stream::build::kv_entries(1, 128), 47);
        assert_eq!(kv_pya(4608, 1, 128), 99);
        assert_eq!(kv_pya(128, 1, 128), 8);
    }
}
