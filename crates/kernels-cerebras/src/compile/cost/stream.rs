//! The stream regime's op-derived step time: each unit's rows priced by what its PEs do a row
//! (matvec tiles at `stream_p1`, region ops at `stream_dsd_elem`, all-reduces, rotations, stores
//! and queries), the units' phases in the order the row stream and the barriers allow. A phase
//! costs its rows x its pace: a row's path through its steps, or one PE's own work a row where
//! that is longer. A pre-barrier phase starts when the rows reach the unit; a post-barrier phase
//! when every unit of its block has stored.

use serde::Serialize;

use crate::compile::stream::build::{Program, kv_pya};
use crate::compile::stream::place;
use std::collections::BTreeMap;

use poem_ir::ValueId;

use crate::compile::stream::unit::{self, Side, Step};
use crate::tuning;

/// A decode step's rotation computed at its root (`srope.csl`'s DIRECT, no tables): a moved
/// position costs [`ROT_DIRECT_OPS`] DSD operations over the pairs, each `stream_rot_direct_op` to
/// issue and `stream_dsd_elem` an element.
pub const ROT_DIRECT_OPS: f64 = 42.0;
/// DIRECT's step (`srope.csl`'s STEP): a position one past the last advanced by
/// [`ROT_STEP_OPS`] DSD operations over the pairs, DIRECT at each multiple of the period.
pub const ROT_STEP_OPS: f64 = 6.0;

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
    /// The rotation tables' bits a digit (`srope.csl`'s `BITS`; 0 without tables).
    pub rope_bits: f64,
    /// A rotation computed at the root (`srope.csl`'s DIRECT): its pairs (`NROT`); 0 with tables.
    pub rope_direct: f64,
    /// DIRECT's anchor period (`srope.csl`'s STEP; 0: every position DIRECT).
    pub rope_step: f64,
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
        if let Some(i) = csl.find("ops/srope.csl") {
            f.rope_bits = csl[i..]
                .split(".BITS = ")
                .nth(1)
                .and_then(|t| t.split(|ch: char| !ch.is_ascii_digit()).next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(0.0);
            if csl[i..].contains(".DIRECT = true") {
                f.rope_direct = csl[i..]
                    .split(".NROT = ")
                    .nth(1)
                    .and_then(|t| t.split(|ch: char| !ch.is_ascii_digit()).next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0.0);
                f.rope_step = csl[i..]
                    .split(".STEP = ")
                    .nth(1)
                    .and_then(|t| t.split(|ch: char| !ch.is_ascii_digit()).next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0.0);
            }
        }
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
        self.cycles_reps(chunk, 1)
    }

    /// The tables' levels recomputed a row (once a row however many calls read the row's
    /// positions: the calls after the first find them held).
    #[must_use]
    pub fn rotation_advance(&self) -> f64 {
        if self.rotations == 0.0 {
            return 0.0;
        }
        if self.rope_direct > 0.0 {
            let op = tuning::get().stream_rot_direct_op
                + self.rope_direct * tuning::get().stream_dsd_elem;
            if self.rope_step > 1.0 {
                let n = self.rope_step;
                return op * (ROT_DIRECT_OPS + (n - 1.0) * ROT_STEP_OPS) / n;
            }
            return ROT_DIRECT_OPS * op;
        }
        let d = self.rope_bits.max(1.0).exp2();
        tuning::get().stream_rot_change + tuning::get().stream_rot_level * d / (d - 1.0)
    }

    /// Cycles a row of `reps` runs of the region over a root's heads (the row's positions
    /// moving the tables once).
    #[must_use]
    pub fn cycles_reps(&self, chunk: u32, reps: u32) -> f64 {
        let levels = f64::from(chunk.max(2)).log2().ceil();
        let one = self.dsd_ops * tuning::get().stream_dsd_op
            + self.dsd_elems * tuning::get().stream_dsd_elem
            + self.exp_elems * tuning::get().stream_exp_elem
            + self.recip_elems * tuning::get().stream_recip_elem
            + self.sum_elems * tuning::get().stream_sum_elem
            + self.sums * levels * tuning::get().stream_sum_level
            + self.scalar_math * tuning::get().stream_scalar_math
            + self.all_reduces * tuning::get().stream_all_reduce
            + self.rotations * tuning::get().stream_rot_call;
        f64::from(reps.max(1)) * one + self.rotation_advance()
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
    /// Each root-side step's cycles a row: (step, `col` / `row` / `store` / `query`, after the
    /// barrier, its roots, its chunk, cycles).
    pub parts: Vec<Part>,
}

/// One root-side step of a unit, priced a row.
#[derive(Clone, Debug, Serialize)]
pub struct Part {
    pub step: usize,
    pub what: &'static str,
    pub late: bool,
    pub roots: u32,
    pub chunk: u32,
    pub cycles: f64,
}

/// The step's priced units and the step's cycles.
#[derive(Clone, Debug, Serialize)]
pub struct StreamCost {
    pub units: Vec<UnitCost>,
    pub cycles: f64,
}

/// A matvec's tile passes a row: two over an f32 input (`smv.csl`'s `X32`: its 16-bit halves).
fn passes(s: &Step) -> f64 {
    match s {
        Step::Matvec { x32: true, .. } => 2.0,
        _ => 1.0,
    }
}

fn region_cycles(p: &Program, u: usize, step: usize) -> f64 {
    let b = &p.units[u];
    let Some((at, printed, params)) = b.regions.get(&step) else {
        return 0.0;
    };
    match &b.helper {
        Some((si, h, hp)) if *si == step => {
            // the root's part, its staging and copies, the helpers' parts out and back
            let (c, q) = (f64::from(at.chunk), f64::from(at.chunk / (h + 1)));
            let np = params
                .iter()
                .filter(|x| **x == crate::compile::emit::region::ParamAt::Row)
                .count() as f64;
            let nout = hp.results.len() as f64;
            RegionFeatures::of(&hp.csl, at.chunk / (h + 1)).cycles(at.chunk / (h + 1))
                + (np * c + np * q + 2.0 * nout * q) * tuning::get().stream_dsd_elem
                + (np + nout) * (c - q) * tuning::get().stream_fabric_word
        }
        _ => RegionFeatures::of(&printed.csl, at.chunk).cycles_reps(at.chunk, at.reps),
    }
}

/// A query's cycles over a K / V column of `rows` stored rows at `pya` attention PEs.
#[must_use]
pub fn query_cycles(rows: u32, pya: u32, q_heads: u32) -> f64 {
    let ne = rows.div_ceil(pya.max(1));
    tuning::get().stream_query + tuning::get().stream_query_entry * f64::from(ne * q_heads)
}

/// The step time a run of `p` is expected to measure: [`time`] by `stream_measured`.
#[must_use]
pub fn expected(p: &Program) -> f64 {
    time(p).cycles * tuning::get().stream_measured
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
        let mut parts = Vec::new();
        // the weight PEs: the busiest column a row, by phase
        for (phase, acc) in [(false, &mut e), (true, &mut lt)] {
            let busiest = (0..l.px())
                .map(|c| {
                    b.unit
                        .steps
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i >= b.unit.split) == phase)
                        .filter_map(|(_, s)| place::tile_dims(s, l, c).map(|t| (t, passes(s))))
                        .map(|((a, bb), x)| f64::from(a * bb) * tuning::get().stream_p1 * x)
                        .sum::<f64>()
                })
                .fold(0.0, f64::max);
            mv = mv.max(busiest);
            // the roots: the row roots' regions a row; a column root's steps a row (the head
            // roots' stores and queries, each region on the roots it runs on)
            let mut row_side = 0.0;
            // a row's path through the phase's steps: a step starts when what it reads has
            // landed (weight PEs and roots finish a row before the next)
            let rd = unit::Reader {
                trace: &p.trace,
                word: b.word,
                prologue: b.unit.classes.is_empty(),
            };
            let mut ready: BTreeMap<u32, f64> = BTreeMap::new();
            let at_of = |ready: &BTreeMap<u32, f64>, v: ValueId| {
                ready
                    .get(&v.0)
                    .or_else(|| ready.get(&unit::unmoved(v).0))
                    .copied()
                    .unwrap_or(0.0)
            };
            let mut path = 0.0f64;
            let mut col_root = vec![0.0f64; l.px().max(1) as usize];
            let mut on = |first: u32, n: u32, c: f64| {
                for r in first..(first + n).min(l.px()) {
                    col_root[r as usize] += c;
                }
            };
            for (i, s) in b.unit.steps.iter().enumerate() {
                if (i >= b.unit.split) != phase {
                    continue;
                }
                let at = b.regions.get(&i).map(|r| r.0);
                let part = |what, cycles| Part {
                    step: i,
                    what,
                    late: phase,
                    roots: at.map_or(l.heads, |a| a.roots),
                    chunk: at.map_or(0, |a| a.chunk),
                    cycles,
                };
                // the step on the row's path
                let (start, cost, made): (f64, f64, Vec<ValueId>) = match s {
                    Step::In { value, .. } => (0.0, 0.0, vec![*value]),
                    Step::Region { chain, .. } => (
                        place::row_params(&rd, chain)
                            .iter()
                            .map(|(_, c)| at_of(&ready, c.base))
                            .fold(0.0, f64::max),
                        region_cycles(p, ui, i),
                        chain
                            .outs
                            .iter()
                            .flat_map(|o| [o.value, rd.cols(o.value).base])
                            .collect(),
                    ),
                    Step::Matvec { input, output, .. } => {
                        // (its busiest column's tile: the sum the next step waits on)
                        let elems = (0..l.px())
                            .filter_map(|c| place::tile_dims(s, l, c))
                            .map(|(a, bb)| f64::from(a * bb))
                            .fold(0.0, f64::max);
                        (
                            at_of(&ready, input.base),
                            elems * tuning::get().stream_p1 * passes(s),
                            vec![*output],
                        )
                    }
                    Step::Store { q, k, v } => (
                        [q.base, k.base, v.base]
                            .iter()
                            .map(|&x| at_of(&ready, x))
                            .fold(0.0, f64::max),
                        tuning::get().stream_store,
                        Vec::new(),
                    ),
                    Step::Attend { out, .. } => (0.0, 0.0, vec![*out]),
                    Step::Out { value, .. } => (at_of(&ready, value.base), 0.0, Vec::new()),
                };
                let mut end = start + cost;
                match s {
                    Step::Region {
                        side: Side::Row, ..
                    } => {
                        let c = region_cycles(p, ui, i);
                        row_side += c;
                        parts.push(part("row", c));
                    }
                    Step::Region {
                        side: Side::Col, ..
                    } => {
                        let c = region_cycles(p, ui, i);
                        let a = at.unwrap_or(place::RegionAt {
                            first: 0,
                            roots: l.px(),
                            chunk: 0,
                            reps: 1,
                            per: 1,
                        });
                        // (a region placed on no roots of its own: on all of them)
                        if a.roots == 0 {
                            on(0, l.px(), c);
                        } else {
                            on(a.first, a.roots, c);
                        }
                        parts.push(part("col", c));
                    }
                    Step::Store { .. } => {
                        on(0, l.heads, tuning::get().stream_store);
                        parts.push(part("store", tuning::get().stream_store));
                    }
                    Step::Attend { .. } => {
                        let block: u32 =
                            p.kv.iter().find(|k| k.units.contains(&ui)).map_or(0, |k| {
                                k.units
                                    .iter()
                                    .map(|&v| rows_of(&p.units[v].unit.classes, class_rows))
                                    .sum()
                            });
                        let pya = p
                            .stream
                            .kv
                            .first()
                            .map_or_else(|| kv_pya(block, qh, a.head_dim), |k| k.pya);
                        let c = query_cycles(block, pya, qh);
                        on(0, l.heads, c);
                        end += c;
                        parts.push(part("query", c));
                    }
                    _ => {}
                }
                for v in made {
                    ready.insert(v.0, end);
                }
                // a row waits on what comes back to its row roots; a dead end on the column side
                // (a store) paces only its own roots
                let back = matches!(
                    s,
                    Step::Region {
                        side: Side::Row,
                        ..
                    } | Step::Matvec {
                        from: Side::Col,
                        ..
                    } | Step::Out {
                        side: Side::Row,
                        ..
                    }
                );
                if back {
                    path = path.max(end);
                }
            }
            if l.spill {
                // the row store: each held chunk cast out before the barrier, summed back up
                // the row chain after it
                let r = crate::compile::stream::unit::Reader {
                    trace: &p.trace,
                    word: b.word,
                    prologue: false,
                };
                let (per, _) = place::store_words(&r, &b.unit, l.py, l.px());
                row_side += f64::from(per) * tuning::get().stream_store_word
                    + if phase {
                        f64::from(l.px()) * tuning::get().stream_chain_hop
                    } else {
                        0.0
                    };
            }
            let col_side = col_root.iter().copied().fold(0.0, f64::max);
            *acc = busiest.max(row_side).max(col_side).max(path);
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
            parts,
        });
    }
    let done = timeline(p, &units, class_rows);
    StreamCost {
        cycles: done,
        units,
    }
}

/// The step's end along the stream, row by row: each unit's CX roots run their schedule's
/// segments in order ([`place::Seg`]), a row no sooner than it reaches the unit and no faster
/// than the unit's pace ([`UnitCost`]), a late segment no sooner than every unit of its block has
/// stored. At `class_rows` other than the build's, each segment's rows scale with its class's.
#[must_use]
pub fn timeline(p: &Program, units: &[UnitCost], class_rows: &[u32]) -> f64 {
    let built: Vec<u32> = p.class_rows.iter().map(|r| r.0).collect();
    let fwd = f64::from(p.stream_hidden() / p.stream.py.max(1)) * tuning::get().stream_forward_word;
    // each path unit's segments with their classes (read off the stream at the build's rows)
    #[derive(Clone, Copy)]
    enum K {
        Early(bool),
        Late,
        Fwd,
        Drop,
    }
    let mut floor = 0.0f64;
    let mut done = 0.0f64;
    let mut plan: Vec<(usize, Vec<(K, u32)>)> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (ui, b) in p.units.iter().enumerate() {
        let Some(sc) = b.sched.as_ref().filter(|_| !b.unit.classes.is_empty()) else {
            // the lane prologue: the rest start once its first row is out
            floor = floor.max(units[ui].early_row);
            done = done.max(f64::from(units[ui].rows) * units[ui].early_row);
            continue;
        };
        let late = sc.segs.iter().any(|g| matches!(g, place::Seg::Late { .. }));
        let class_of = |t0: u32| sc.rows.get(t0 as usize).map_or(0, |r| r.0);
        let scale = |c: usize, n: u32| -> u32 {
            match (built.get(c), class_rows.get(c)) {
                (Some(&b0), Some(&r)) if b0 > 0 => {
                    (u64::from(n) * u64::from(r)).div_ceil(u64::from(b0)) as u32
                }
                _ => n,
            }
        };
        let (mut idx, mut out, mut segs) = (0usize, Vec::new(), Vec::new());
        for g in &sc.segs {
            match *g {
                place::Seg::Early { t0, n, host } => {
                    let c = class_of(t0);
                    if !host {
                        idx += n as usize;
                    }
                    if !late {
                        out.extend(std::iter::repeat_n(c, n as usize));
                    }
                    segs.push((K::Early(host), scale(c, n)));
                }
                place::Seg::Late { t0, n } => {
                    let c = class_of(t0);
                    out.extend(std::iter::repeat_n(c, n as usize));
                    segs.push((K::Late, scale(c, n)));
                }
                place::Seg::Fwd { n } | place::Seg::Drop { n } => {
                    let c = labels.get(idx).copied().unwrap_or(0);
                    if matches!(g, place::Seg::Fwd { .. }) {
                        out.extend(labels.iter().skip(idx).take(n as usize));
                        segs.push((K::Fwd, scale(c, n)));
                    } else {
                        segs.push((K::Drop, scale(c, n)));
                    }
                    idx += n as usize;
                }
            }
        }
        labels = out;
        plan.push((ui, segs));
    }
    let block_of = |u: usize| p.kv.iter().position(|k| k.units.contains(&u));
    let mut barrier = vec![0.0f64; p.kv.len()];
    for _ in 0..16 {
        let mut stored = vec![0.0f64; p.kv.len()];
        let mut inp: Vec<f64> = Vec::new();
        let mut end = done;
        for (ui, segs) in &plan {
            let c = &units[*ui];
            let (mut clock, mut idx, mut out) = (floor, 0usize, Vec::new());
            let arrive = |idx: &mut usize| -> f64 {
                let t = inp.get(*idx).or(inp.last()).copied().unwrap_or(floor);
                *idx += 1;
                t
            };
            let mut early_end = floor;
            let late = segs.iter().any(|s| matches!(s.0, K::Late));
            for &(k, n) in segs {
                for _ in 0..n {
                    match k {
                        K::Early(host) => {
                            if !host {
                                clock = clock.max(arrive(&mut idx));
                            }
                            clock += c.early_row;
                            if !late {
                                out.push(clock);
                            }
                        }
                        K::Late => {
                            clock = clock.max(block_of(*ui).map_or(0.0, |b| barrier[b]));
                            clock += c.late_row;
                            out.push(clock);
                        }
                        K::Fwd => {
                            clock = clock.max(arrive(&mut idx)) + fwd;
                            out.push(clock);
                        }
                        K::Drop => clock = clock.max(arrive(&mut idx)) + fwd,
                    }
                }
                if matches!(k, K::Early(_)) {
                    early_end = clock;
                }
            }
            if let Some(b) = block_of(*ui) {
                stored[b] = stored[b].max(early_end);
            }
            end = end.max(clock);
            inp = out;
        }
        let settled = stored == barrier;
        barrier = stored;
        if settled {
            return end;
        }
    }
    done
}
