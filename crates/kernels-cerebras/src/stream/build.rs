//! A stream reading's wafer program, end to end before the CSL: its units (the lane prologue's,
//! then a unit a side of every stage in path order: [`super::unit`]), their layouts and the row
//! stream's schedule ([`super::place`]), the floorplan's elements ([`crate::floorplan::stream`])
//! and the fabric they are laid on (coloured by [`Palette::allocate`]).

use std::collections::{BTreeMap, BTreeSet};

use poem_compiler::spatial::layer::stream::{HostKind, Stage};
use poem_ir::{Def, Operands, Trace, ValueId};

use super::StreamShape;
use super::place::{self, KvSched, Layout, UnitSched};
use super::unit::{self, Reader, Side, Step, Unit};
use crate::emit::region::{self, ParamAt, Place, Printed};
use crate::fabric::{Fabric, Palette};
use crate::floorplan::Requests;
use crate::floorplan::stream::{Bounds, Stream, UnitAsk};

type R<T> = Result<T, String>;

/// Attention PEs a K / V column at least.
pub const PYA: u32 = 8;
/// 16-bit words an attention PE gives its entries (of 24,576: the rest is code and buffers).
pub const KV_WORDS: u32 = 18_432;

/// Entries an attention PE keeps at most (ops/satt.csl with QH): a key, a value and its ones
/// column, the held q, and the scores' f32, 16-bit and probability words a q head.
#[must_use]
pub fn kv_entries(q_heads: u32, head_dim: u32) -> u32 {
    let per = head_dim + (head_dim + 1) + q_heads * head_dim + q_heads * 6;
    (KV_WORDS / per).max(1)
}

/// The attention PEs a K / V column of `rows` stored rows takes (P7 v1: a column as long as its
/// entries ask, rows `p` kept by PE `p % pya`).
#[must_use]
pub fn kv_pya(rows: u32, q_heads: u32, head_dim: u32) -> u32 {
    rows.div_ceil(kv_entries(q_heads, head_dim)).max(PYA)
}

/// A unit as built: its steps, layout, printed regions (by step index) and place on the stream.
#[derive(Clone, Debug)]
pub struct Built {
    pub unit: Unit,
    /// The class word its nodes are read for.
    pub word: u64,
    pub layout: Layout,
    pub regions: BTreeMap<usize, (place::RegionAt, Printed, Vec<ParamAt>)>,
    /// Its schedule on the row stream (`None`: the lane prologue).
    pub sched: Option<UnitSched>,
    /// It reads lane vectors on its row / column roots.
    pub lanes: (bool, bool),
}

/// The whole program.
#[derive(Clone, Debug)]
pub struct Program {
    pub trace: Trace,
    pub shape: StreamShape,
    pub units: Vec<Built>,
    pub kv: Vec<KvSched>,
    pub stream: Stream,
    pub palette: Palette,
    pub fabric: Fabric,
    /// The rows each class brings, its first row in the step's order.
    pub class_rows: Vec<(u32, u32)>,
    /// Bus words a step: the lane prologue's outputs, in its CY roots' order.
    pub lane_words: u32,
    /// Each lane base's words on the bus: (base, from) -> bus offset, run length.
    pub lane_at: Vec<(u32, u32, u32, u32)>,
    pub round16: bool,
}

impl Program {
    #[must_use]
    pub fn shape_trace(&self) -> &Trace {
        &self.trace
    }

    /// The row stream's width (the residual's).
    #[must_use]
    pub fn stream_hidden(&self) -> u32 {
        self.shape.hidden
    }

    /// The bus offset of lane value `base`'s column `col`.
    #[must_use]
    pub fn lane_offset(&self, base: ValueId, col: u32) -> Option<u32> {
        self.lane_at
            .iter()
            .find(|&&(b, f, _, n)| b == base.0 && col >= f && col < f + n)
            .map(|&(_, f, o, _)| o + col - f)
    }
}

/// `taps`: values a unit makes it also leaves to the host, a row at a time, right where it
/// makes them (a debugging read of a step's insides on the simulator).
fn tap(trace: &Trace, u: &mut Unit, taps: &[u32]) {
    let r = Reader {
        trace,
        word: 0,
        prologue: false,
    };
    let mut i = 0;
    while i < u.steps.len() {
        let made = unit::step_makes(&u.steps[i]);
        if let Some(&v) = made.iter().find(|v| taps.contains(&v.0)) {
            let side = u.sides[&v];
            u.steps.insert(
                i + 1,
                Step::Out {
                    value: unit::Cols {
                        base: v,
                        from: 0,
                        width: r.width(v),
                    },
                    side,
                    host: true,
                    lane: false,
                },
            );
            if i < u.split {
                u.split += 1;
            }
            i += 1;
        }
        i += 1;
    }
}

/// Whether `s` reads a lane vector (a resident picked by the row's lane).
fn reads_lanes(r: &Reader, s: &Step) -> bool {
    match s {
        Step::Region { chain, .. } => chain
            .ins
            .iter()
            .any(|i| i.by.is_some() || (r.lane(i.value) && !i.shared)),
        _ => false,
    }
}

/// The program of `trace`'s stream `shape` (bf16 roundings at the regions' points where
/// `round16`).
pub fn program(trace: &Trace, shape: StreamShape, round16: bool) -> R<Program> {
    program_upto(trace, shape, round16, None, &[])
}

/// [`program`] over the first `upto` stages only (a part of a step on the simulator): the last
/// stage's units leave their rows to the host. The lane prologue is left out where no unit kept
/// reads a lane vector.
pub fn program_upto(
    trace: &Trace,
    shape: StreamShape,
    round16: bool,
    upto: Option<usize>,
    taps: &[u32],
) -> R<Program> {
    program_py(trace, shape, round16, upto, taps, None)
}

/// The row roots a unit may take: the divisors of the hidden width that leave a CX root at
/// least [`MIN_CHUNK`] columns and a band within the fabric's height.
#[must_use]
pub fn py_candidates(hidden: u32, bounds: Bounds) -> Vec<u32> {
    (1..=hidden / MIN_CHUNK)
        .filter(|&py| hidden.is_multiple_of(py) && py >= MIN_PY && py + 2 + PYA < bounds.h)
        .collect()
}

/// A CX root's fewest columns of the row stream, a unit's fewest row roots.
pub const MIN_CHUNK: u32 = 4;
pub const MIN_PY: u32 = 4;

/// [`program_upto`] at `py` row roots a unit, or (`None`) at the row roots the cost model
/// prices lowest of those that build within the system fabric ([`py_candidates`]).
pub fn program_py(
    trace: &Trace,
    shape: StreamShape,
    round16: bool,
    upto: Option<usize>,
    taps: &[u32],
    py: Option<u32>,
) -> R<Program> {
    let bounds = Bounds::system();
    let hidden = shape.program.hidden;
    let cands = match py {
        Some(p) => vec![p],
        None => py_candidates(hidden, bounds),
    };
    let tried: Vec<(u32, R<Program>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = cands
            .iter()
            .map(|&p| {
                let shape = shape.clone();
                sc.spawn(move || (p, program_at(trace, shape, round16, upto, taps, p, bounds)))
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().unwrap_or_else(|_| (0, Err("panicked".into()))))
            .collect()
    });
    let mut best: Option<(f64, Program)> = None;
    let mut errs = Vec::new();
    for (p, r) in tried {
        match r {
            Ok(prog) => {
                let c = crate::cost::stream::time(&prog).cycles;
                if best.as_ref().is_none_or(|(b, _)| c < *b) {
                    best = Some((c, prog));
                }
            }
            Err(e) => errs.push(format!("py {p}: {e}")),
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        if errs.is_empty() {
            format!(
                "{}: a hidden width of {hidden} cuts over no rows",
                trace.name
            )
        } else {
            errs.join("; ")
        }
    })
}

/// [`program_upto`] at `py` row roots a unit within `bounds`.
pub fn program_at(
    trace: &Trace,
    shape: StreamShape,
    round16: bool,
    upto: Option<usize>,
    taps: &[u32],
    py: u32,
    bounds: Bounds,
) -> R<Program> {
    let prog = &shape.program;
    let name = &trace.name;
    if py == 0 || !prog.hidden.is_multiple_of(py) {
        return Err(format!(
            "{name}: a hidden width of {} cuts over no {py} rows",
            prog.hidden
        ));
    }
    // the host's nodes: its permutations and tables, and the lane ops it computes from its own
    // inputs alone (a sinusoid of the timestep)
    let mut host: BTreeSet<u32> = BTreeSet::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        let _ = node;
        let _ = at;
    }
    {
        // the stream program names its host ops by op; read them off the nodes again
        let names: Vec<&str> = prog.host.iter().map(|h| h.op.op).collect();
        let kinds: Vec<HostKind> = prog.host.iter().map(|h| h.kind).collect();
        let _ = kinds;
        let mut ins = Vec::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            let op = node.op.name();
            if !names.contains(&op) || prog.lane_nodes.contains(&(at as u32)) {
                continue;
            }
            if prog
                .stages
                .iter()
                .any(|s| s.nodes.iter().any(|n| n.contains(&(at as u32))))
            {
                continue;
            }
            host.insert(at as u32);
        }
        for &at in &prog.lane_nodes {
            let node = &trace.nodes[at as usize];
            ins.clear();
            node.op.inputs(&mut ins);
            let own = ins.iter().all(|v| {
                matches!(
                    trace.values.get(v.0 as usize).map(|d| &d.def),
                    Some(Def::Input(_))
                )
            });
            if own {
                host.insert(at);
            }
        }
    }
    let class_rows: Vec<(u32, u32)> = {
        let mut at = 0;
        prog.classes
            .iter()
            .map(|c| {
                let r = (c.rows, at);
                at += c.rows;
                r
            })
            .collect()
    };

    // the units: the lane prologue, then each stage's sides in order
    let mut units: Vec<(Unit, u64)> = Vec::new();
    let lane_nodes: Vec<u32> = prog
        .lane_nodes
        .iter()
        .copied()
        .filter(|n| !host.contains(n))
        .collect();
    let any_word = prog.classes.first().map_or(0, |c| c.word);
    if !lane_nodes.is_empty() {
        let words: Vec<u64> = prog.classes.iter().map(|c| c.word).collect();
        let u = unit::lower(trace, &words, "lanes", &lane_nodes, &host, 1, true)?;
        units.push((u, any_word));
    }
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let kept = upto.unwrap_or(prog.stages.len()).min(prog.stages.len());
    for (si, st) in prog.stages[..kept].iter().enumerate() {
        let tag = match st.stage {
            Stage::Entry => "entry".to_string(),
            Stage::Layer { stack, layer } => format!("s{stack}l{layer}"),
            Stage::Between { stack } => format!("after{stack}"),
            Stage::Exit => "exit".to_string(),
        };
        let mut blk = Vec::new();
        for (side, nodes) in st.sides.iter().zip(&st.nodes) {
            let word = prog.classes[side.classes[0]].word;
            let cname: Vec<&str> = side
                .classes
                .iter()
                .map(|&c| prog.classes[c].name.as_str())
                .collect();
            let rows = side.classes.iter().map(|&c| prog.classes[c].rows).sum();
            let mut u = unit::lower(
                trace,
                &[word],
                &format!("{tag}.{}", cname.join("+")),
                nodes,
                &host,
                rows,
                false,
            )?;
            u.classes = side.classes.clone();
            tap(trace, &mut u, taps);
            if si + 1 == kept && kept < prog.stages.len() {
                for s in &mut u.steps {
                    if let Step::Out { host, .. } = s {
                        *host = true;
                    }
                }
            }
            if u.joint.is_some() {
                blk.push(units.len());
            }
            units.push((u, word));
        }
        if !blk.is_empty() {
            blocks.push(blk);
        }
    }

    // layouts and regions
    let mut built: Vec<Built> = Vec::new();
    for (u, word) in units {
        let r = Reader {
            trace,
            word,
            prologue: u.classes.is_empty(),
        };
        let layout = place::layout(&r, &u, py)?;
        let mut regions = BTreeMap::new();
        for (i, s) in u.steps.iter().enumerate() {
            let Step::Region { side, chain } = s else {
                continue;
            };
            let at = layout.region_at(*side, chain.region.width, &place::region_cols(&r, chain));
            let params: Vec<ParamAt> = chain
                .ins
                .iter()
                .enumerate()
                .map(|(k, input)| {
                    if chain.region.params[k] == kernels_ir::Shape::Scalar {
                        ParamAt::Scalar
                    } else if input.shared || input.by.is_some() || r.lane(input.value) {
                        ParamAt::Resident
                    } else {
                        ParamAt::Row
                    }
                })
                .collect();
            // positions below a class's rows: digits of 3 bits
            let top = class_rows.iter().map(|c| c.0).max().unwrap_or(1).max(2) - 1;
            let lv = (32 - top.leading_zeros()).div_ceil(3);
            let printed = region::print(
                &chain.region,
                &Place {
                    chunk: at.chunk,
                    roots: at.roots,
                    params: params.clone(),
                    round16,
                    rope_lv: lv.max(1),
                    rope_bits: 3,
                },
            )
            .map_err(|e| format!("{}: step {i}: {e}", u.name))?;
            regions.insert(i, (at, printed, params));
        }
        let lanes = (
            u.steps.iter().any(|s| {
                matches!(
                    s,
                    Step::Region {
                        side: Side::Row,
                        ..
                    }
                ) && reads_lanes(&r, s)
            }),
            u.steps.iter().any(|s| {
                matches!(
                    s,
                    Step::Region {
                        side: Side::Col,
                        ..
                    }
                ) && reads_lanes(&r, s)
            }),
        );
        built.push(Built {
            unit: u,
            word,
            layout,
            regions,
            sched: None,
            lanes,
        });
    }

    // the row stream
    if built.first().is_some_and(|b| b.unit.classes.is_empty())
        && !built.iter().any(|b| b.lanes.0 || b.lanes.1)
    {
        built.remove(0);
    }
    let lane_unit = built.first().is_some_and(|b| b.unit.classes.is_empty());
    let path: Vec<usize> = (0..built.len())
        .filter(|&i| !built[i].unit.classes.is_empty())
        .collect();
    let path_units: Vec<(Vec<usize>, bool, bool, bool)> = path
        .iter()
        .map(|&i| {
            let u = &built[i].unit;
            (
                u.classes.clone(),
                u.steps
                    .iter()
                    .any(|s| matches!(s, Step::In { host: true, .. })),
                u.joint.is_some(),
                u.steps
                    .iter()
                    .any(|s| matches!(s, Step::Out { host: true, .. })),
            )
        })
        .collect();
    let rows_of: Vec<u32> = class_rows.iter().map(|r| r.0).collect();
    let path_blocks: Vec<Vec<usize>> = blocks
        .iter()
        .map(|b| {
            b.iter()
                .map(|u| path.iter().position(|p| p == u).unwrap_or(0))
                .collect()
        })
        .collect();
    let (scheds, kv) = place::schedule(&path_units, &rows_of, &path_blocks)?;
    for (k, s) in scheds.into_iter().enumerate() {
        built[path[k]].sched = Some(s);
    }
    let kv: Vec<KvSched> = kv
        .into_iter()
        .map(|mut k| {
            k.units = k.units.iter().map(|&p| path[p]).collect();
            k
        })
        .collect();

    // the lane bus's words: the lane prologue's CY roots' lane outputs, in root order
    let mut lane_at = Vec::new();
    let mut lane_words = 0;
    if lane_unit {
        let b = &built[0];
        for (i, col) in b.layout.cols.iter().enumerate() {
            for s in col {
                let out = b.unit.steps.iter().any(|st| {
                    matches!(st, Step::Out { value, lane: true, .. } if value.base.0 == s.base)
                });
                if out {
                    lane_at.push((s.base, s.from, lane_words, s.len));
                    lane_words += s.len;
                }
            }
            let _ = i;
        }
    }

    // the floorplan
    let heads = shape.attn.kv_heads;
    let requests = Requests {
        stream: true,
        joint: !blocks.is_empty(),
        lanes: lane_unit,
        ..Requests::default()
    };
    let asks: Vec<UnitAsk> = built
        .iter()
        .map(|b| UnitAsk {
            name: b.unit.name.clone(),
            heads: b.layout.heads,
            cols: b.layout.px(),
            segment: b
                .regions
                .values()
                .any(|(at, p, _)| at.roots == py && p.all_reduces > 0 && at.first == 0)
                && b.unit.steps.iter().any(|s| {
                    matches!(
                        s,
                        Step::Region {
                            side: Side::Row,
                            ..
                        }
                    )
                }),
            on_path: !b.unit.classes.is_empty(),
            lanes: b.lanes,
        })
        .collect();
    let qh = shape.attn.q_heads / shape.attn.kv_heads.max(1);
    let pya = kv
        .iter()
        .map(|k| kv_pya(k.stores.iter().map(|s| s.1).sum(), qh, shape.attn.head_dim))
        .max()
        .unwrap_or(PYA);
    let stream = Stream::of(&requests, py, &asks, &blocks, heads, pya, bounds)?;
    let wiring = stream.wiring();
    let floor = stream.floor();
    let palette = Palette::allocate(Vec::new(), &floor, &wiring)?;
    let fabric = Fabric::lay(&floor, &wiring, &palette)?;
    fabric.validate(&floor)?;
    Ok(Program {
        trace: trace.clone(),
        shape,
        units: built,
        kv,
        stream,
        palette,
        fabric,
        class_rows,
        lane_words,
        lane_at,
        round16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace() -> Trace {
        poem_compiler::catalog::repository()
            .parse("flux2-mini-bf16-kv-bf16")
            .expect("flux2-mini parses")
            .trace(poem_ir::Platform::Cerebras)
    }

    /// flux2-mini's denoise builds: the lane prologue, 2 entries, 4 double sides, 2 single
    /// blocks and the exit, a K / V column a head under each of the 4 attending blocks, coloured
    /// in the fabric's 21.
    #[test]
    fn flux_2_minis_denoise_builds_a_stream_program() {
        let t = trace();
        let s = super::super::shape_of(&t, "denoise", 64).unwrap_or_else(|e| panic!("{e}"));
        let p = program(&t, s, true).unwrap_or_else(|e| panic!("{e}"));
        for b in &p.units {
            eprintln!(
                "{}: {} steps (split {}), {} x {} roots, heads {}, held {:?}",
                b.unit.name,
                b.unit.steps.len(),
                b.unit.split,
                p.stream.py,
                b.layout.px(),
                b.layout.heads,
                b.unit.held
            );
        }
        eprintln!("fabric {} x {}", p.stream.w, p.stream.h);
        assert_eq!(p.units.len(), 10);
        assert_eq!(p.kv.len(), 4);
        assert_eq!(p.stream.kv.len(), 8);
        assert_eq!(p.lane_words, 1536 + 1536 + 768 + 512);
    }
}
