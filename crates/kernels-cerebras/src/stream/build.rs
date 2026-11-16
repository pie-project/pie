//! A stream reading's wafer program, end to end before the CSL: its units (the lane prologue's,
//! then a unit a side of every stage in path order: [`super::unit`]), their layouts and the row
//! stream's schedule ([`super::place`]), the floorplan's elements ([`crate::floorplan::stream`])
//! and the fabric they are laid on (coloured by [`Palette::allocate`]).

use std::collections::{BTreeMap, BTreeSet};

use poem_compiler::spatial::layer::stream::{Stage, StreamProgram};
use poem_ir::{Def, Linear, Operands, Operation, Trace, ValueId};

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
/// column, the held q, and the scores' f32, 16-bit and probability words a q head; less what
/// its row buffers take past one q head's (a kv head of several q heads: grouped-query).
#[must_use]
pub fn kv_entries(q_heads: u32, head_dim: u32) -> u32 {
    let per = head_dim + (head_dim + 1) + q_heads * head_dim + q_heads * 6;
    // 16-bit words of satt's per-query buffers (xh, y, sx, qf) at `q` q heads a kv head
    let rows =
        |q: u32| q * head_dim + 2 * q * (head_dim + 1) + 2 * (q + 2) * head_dim + 2 * q * head_dim;
    let extra = rows(q_heads.max(1)) - rows(1);
    (KV_WORDS.saturating_sub(extra) / per).max(1)
}

/// The attention PEs a K / V column of `rows` stored rows takes (P7 v1: a column as long as its
/// entries ask, rows `p` kept by PE `p % pya`).
#[must_use]
pub fn kv_pya(rows: u32, q_heads: u32, head_dim: u32) -> u32 {
    kv_pya_split(rows, q_heads, head_dim, KvSplit::default())
}

/// How a program's roots and K / V columns are cut past the fewest that fit: a K / V column's
/// PEs keep at most `1 / thin` of the entries a PE can hold (more PEs a head, a query's work
/// over more of them), over `sub` columns a head; a unit whose column regions run on its rest
/// roots takes `widen` quarters more of them than the fewest its tiles fit (smaller chunks: the
/// spare fabric spent on the regions).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct KvSplit {
    pub thin: u32,
    pub sub: u32,
    pub widen: u32,
    /// Helper PEs a rest root at most for its costliest column region ([`Built::helper`]).
    pub help: u32,
}

impl Default for KvSplit {
    fn default() -> KvSplit {
        KvSplit {
            thin: 1,
            sub: 1,
            widen: 0,
            help: 0,
        }
    }
}

/// The cuts the search tries: the fewest rest roots, a quarter and a half more of them; helpers
/// below the rest roots; K / V columns over `sub` columns a head at `1 / sub` of the entries a
/// PE (as many rows of the band, `sub` times the PEs: a query's work over them), the head
/// columns of a block's first unit `sub` apart ([`crate::floorplan::stream::KvColumn`]).
pub const KV_SPLITS: [KvSplit; 11] = [
    KvSplit {
        thin: 1,
        sub: 1,
        widen: 0,
        help: 0,
    },
    KvSplit {
        thin: 1,
        sub: 1,
        widen: 1,
        help: 0,
    },
    KvSplit {
        thin: 1,
        sub: 1,
        widen: 2,
        help: 0,
    },
    KvSplit {
        thin: 1,
        sub: 1,
        widen: 0,
        help: 3,
    },
    KvSplit {
        thin: 1,
        sub: 1,
        widen: 0,
        help: 7,
    },
    KvSplit {
        thin: 1,
        sub: 1,
        widen: 0,
        help: 15,
    },
    KvSplit {
        thin: 2,
        sub: 2,
        widen: 0,
        help: 0,
    },
    KvSplit {
        thin: 2,
        sub: 2,
        widen: 0,
        help: 7,
    },
    KvSplit {
        thin: 2,
        sub: 2,
        widen: 0,
        help: 15,
    },
    KvSplit {
        thin: 2,
        sub: 2,
        widen: 0,
        help: 23,
    },
    KvSplit {
        thin: 3,
        sub: 3,
        widen: 0,
        help: 15,
    },
];

/// [`kv_pya`] at a cut.
#[must_use]
pub fn kv_pya_split(rows: u32, q_heads: u32, head_dim: u32, kv: KvSplit) -> u32 {
    let per = (kv_entries(q_heads, head_dim) / kv.thin.max(1)).max(1);
    rows.div_ceil(per).max(PYA)
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
    /// A column region on its rest roots split over helper PEs below each root: (step, helpers
    /// a root, the region printed at the chunk each PE computes, `chunk / (helpers + 1)`).
    pub helper: Option<(usize, u32, Printed)>,
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

/// What each of `units` (the stream's, in path order; the first unit index `first`) must
/// carry on the row stream past itself: a value it takes in, or that reaches it from the unit
/// before it of its classes, that a later unit of its classes takes in and that it does not
/// send on itself. (Rows of other classes pass a unit by the schedule's forwarding.)
fn carried(units: &[(Unit, u64)], first: usize) -> BTreeMap<usize, Vec<ValueId>> {
    let ins = |u: &Unit| -> Vec<ValueId> {
        u.steps
            .iter()
            .filter_map(|s| match s {
                Step::In {
                    value,
                    side: Side::Row,
                    host: false,
                } => Some(unit::unmoved(*value)),
                _ => None,
            })
            .collect()
    };
    let outs = |u: &Unit| -> Vec<ValueId> {
        u.steps
            .iter()
            .filter_map(|s| match s {
                Step::Out {
                    value,
                    side: Side::Row,
                    host: false,
                    lane: false,
                } => Some(unit::unmoved(value.base)),
                _ => None,
            })
            .collect()
    };
    let mut out = BTreeMap::new();
    let mut arriving: BTreeMap<Vec<usize>, Vec<ValueId>> = BTreeMap::new();
    for (i, (u, _)) in units.iter().enumerate() {
        let later: Vec<ValueId> = units[i + 1..]
            .iter()
            .filter(|(v, _)| v.classes == u.classes)
            .flat_map(|(v, _)| ins(v))
            .collect();
        let sent = outs(u);
        let mut have = arriving.remove(&u.classes).unwrap_or_default();
        have.extend(ins(u));
        let mut thr: Vec<ValueId> = Vec::new();
        for v in have {
            if later.contains(&v) && !sent.contains(&v) && !thr.contains(&v) {
                thr.push(v);
            }
        }
        let mut next = sent;
        next.extend(thr.iter().copied());
        arriving.insert(u.classes.clone(), next);
        if !thr.is_empty() {
            out.insert(first + i, thr);
        }
    }
    out
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
    program_py(
        trace,
        shape,
        round16,
        upto,
        taps,
        None,
        None,
        None,
        false,
        Bounds::system(),
    )
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

/// The tile budgets a layout is tried at: a weight PE's whole budget, and less (more column
/// roots a unit, the spare fabric spent on shorter rows).
pub const TILE_FRACTIONS: [(u32, u32); 3] = [(1, 1), (3, 4), (1, 2)];

/// [`program_upto`] at `py` row roots a unit and `tile` words a weight PE (`spill`: rows held over
/// the barrier in the row store always), or (`None`) at the
/// (row roots, tile budget) the
/// cost model prices lowest of those that build within `bounds` ([`py_candidates`],
/// [`TILE_FRACTIONS`]; [`Bounds::system`] but in tests of a narrower fabric).
#[allow(clippy::too_many_arguments)]
pub fn program_py(
    trace: &Trace,
    shape: StreamShape,
    round16: bool,
    upto: Option<usize>,
    taps: &[u32],
    py: Option<u32>,
    tile: Option<u32>,
    kv: Option<KvSplit>,
    spill: bool,
    bounds: Bounds,
) -> R<Program> {
    // (row roots and tiles both pinned: the program as pinned, its cut the plain one unless
    // pinned too; flux2-mini's `--py 16 --tile 12288` is M1d's full step)
    let kvs: Vec<KvSplit> = match kv {
        Some(k) => vec![k],
        None if py.is_some() && tile.is_some() => vec![KvSplit::default()],
        None => KV_SPLITS.to_vec(),
    };
    let hidden = shape.program.hidden;
    let pys = match py {
        Some(p) => vec![p],
        None => py_candidates(hidden, bounds),
    };
    let tiles: Vec<u32> = match tile {
        Some(t) => vec![t],
        None => TILE_FRACTIONS
            .iter()
            .map(|&(a, b)| place::TILE_BUDGET * a / b)
            .collect(),
    };
    let mut cands: Vec<(u32, u32, KvSplit)> = Vec::new();
    for &p in &pys {
        for &t in &tiles {
            cands.extend(kvs.iter().map(|&k| (p, t, k)));
        }
    }
    // built a few at a time (a whole-wafer program is large), the cheapest kept
    let mut best: Option<(f64, Program)> = None;
    let mut errs: Vec<(u32, u32, String)> = Vec::new();
    for group in cands.chunks(8) {
        let tried: Vec<(u32, u32, R<Program>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = group
                .iter()
                .map(|&(p, tile, kv)| {
                    let shape = shape.clone();
                    sc.spawn(move || {
                        let r = program_at(
                            trace, shape, round16, upto, taps, p, tile, kv, spill, bounds,
                        );
                        (p, tile, r)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or_else(|_| (0, 0, Err("panicked".into()))))
                .collect()
        });
        for (p, tile, r) in tried {
            match r {
                Ok(prog) => {
                    let c = crate::cost::stream::time(&prog).cycles;
                    if best.as_ref().is_none_or(|(b, _)| c < *b) {
                        best = Some((c, prog));
                    }
                }
                Err(e) => errs.push((p, tile, e)),
            }
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        // the farthest a candidate got: the most row roots at the whole tile budget
        match errs
            .iter()
            .filter(|e| e.1 == place::TILE_BUDGET)
            .max_by_key(|e| e.0)
            .or(errs.last())
        {
            Some((p, _, e)) => format!(
                "{}: none of {} (row roots, tile budget) builds within {} x {}; at py {p}: {e}",
                trace.name,
                cands.len(),
                bounds.w,
                bounds.h
            ),
            None => format!(
                "{}: a hidden width of {hidden} cuts over no rows",
                trace.name
            ),
        }
    })
}

/// What a stream reading lowers to before any placement: the host's nodes, each class's rows
/// and first row in the step, and the units (the lane prologue, then each stage's sides) with
/// the class word each is read for, and the attending units of each block. Every regime that
/// runs a stream program reads it (the row stream's units, [`program_at`]; the GEMM regime's
/// whole-fabric ops, `stream::gemm`).
pub struct Lowered {
    pub host: BTreeSet<u32>,
    pub class_rows: Vec<(u32, u32)>,
    pub units: Vec<(Unit, u64)>,
    pub blocks: Vec<Vec<usize>>,
}

/// [`Lowered`] of `prog` (`trace`'s stream program), its first `upto` stages (the last's units
/// leaving their rows to the host), `taps` also read out.
pub fn lower_units(
    trace: &Trace,
    prog: &StreamProgram,
    upto: Option<usize>,
    taps: &[u32],
) -> R<Lowered> {
    let name = &trace.name;
    // the host's nodes: the stream program's (its permutations and tables), and the lane ops it
    // computes from its own inputs, the checkpoint's constants and its other lane ops alone (a
    // sinusoid of the timestep, the timestep's affine map before it), a matvec aside
    let mut host: BTreeSet<u32> = prog.host.iter().map(|h| h.node).collect();
    {
        let mut ins = Vec::new();
        for &at in &prog.lane_nodes {
            let node = &trace.nodes[at as usize];
            if matches!(node.op, Operation::Linear(Linear::Matmul { .. })) {
                continue;
            }
            ins.clear();
            node.op.inputs(&mut ins);
            let own = ins
                .iter()
                .all(|v| match trace.values.get(v.0 as usize).map(|d| &d.def) {
                    Some(Def::Input(_) | Def::Weight(_)) => true,
                    Some(Def::Op(n)) => host.contains(n),
                    _ => false,
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
    // the stages' units, again with what each carries past itself where a later unit reads a
    // value the row stream must bring it (an encoder's earlier tapped layers)
    let first = units.len();
    let mut through: BTreeMap<usize, Vec<ValueId>> = BTreeMap::new();
    for pass in 0..2 {
        units.truncate(first);
        blocks.clear();
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
                // a side of several classes leaves what any of them reads after it (the last
                // single block's rows reach the exit by the image class alone)
                let words: Vec<u64> = side.classes.iter().map(|&c| prog.classes[c].word).collect();
                let cname: Vec<&str> = side
                    .classes
                    .iter()
                    .map(|&c| prog.classes[c].name.as_str())
                    .collect();
                let rows = side.classes.iter().map(|&c| prog.classes[c].rows).sum();
                let mut u = unit::lower_through(
                    trace,
                    &words,
                    &format!("{tag}.{}", cname.join("+")),
                    nodes,
                    &host,
                    rows,
                    false,
                    through.get(&units.len()).map_or(&[][..], |v| &v[..]),
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
        if pass > 0 {
            break;
        }
        through = carried(&units[first..], first);
        if through.is_empty() {
            break;
        }
        if prog.classes.len() > 1 {
            return Err(format!(
                "{name}: the row stream carries a value past a unit in a step of several row classes"
            ));
        }
    }

    Ok(Lowered {
        host,
        class_rows,
        units,
        blocks,
    })
}

/// [`program_upto`] at `py` row roots a unit and weight tiles of at most `tile` words, within
/// `bounds` (`spill`: every unit's rows held over the barrier in its row store).
#[allow(clippy::too_many_arguments)]
pub fn program_at(
    trace: &Trace,
    shape: StreamShape,
    round16: bool,
    upto: Option<usize>,
    taps: &[u32],
    py: u32,
    tile: u32,
    split: KvSplit,
    spill: bool,
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
    let Lowered {
        class_rows,
        units,
        blocks,
        ..
    } = lower_units(trace, prog, upto, taps)?;

    // layouts and regions
    let mut built: Vec<Built> = Vec::new();
    for (u, word) in units {
        let r = Reader {
            trace,
            word,
            prologue: u.classes.is_empty(),
        };
        // a row-side value its row roots hold padded (narrower than them, or of no multiple)
        // passes through matvecs and the row stream; a row region reads it whole
        for st in &u.steps {
            if let Step::Region {
                side: Side::Row,
                chain,
            } = st
                && !chain.region.width.is_multiple_of(py)
            {
                return Err(format!(
                    "{name}: {}: a row region of {} columns cuts over no {py} rows",
                    u.name, chain.region.width
                ));
            }
        }
        let layout = place::layout(&r, &u, py, tile, spill, split.widen)?;
        let mut regions = BTreeMap::new();
        for (i, s) in u.steps.iter().enumerate() {
            let Step::Region { side, chain } = s else {
                continue;
            };
            let at = layout.region_of(*side, &chain.region, &place::region_cols(&r, chain));
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
                    // a head region is whole at each root it runs at
                    roots: if at.reps > 1 || at.chunk == chain.region.width && at.roots > 1 {
                        1
                    } else {
                        at.roots
                    },
                    params: params.clone(),
                    round16,
                    rope_lv: lv.max(1),
                    rope_bits: 3,
                    straddle: false,
                },
            )
            .map_err(|e| format!("{}: step {i}: {e}", u.name))?;
            regions.insert(i, (at, printed, params));
        }
        // helpers: the costliest column region on the rest roots that splits evenly (rows and
        // scalars only: no tables, reductions or rotations), at most `help` PEs a root
        let mut helper = None;
        if split.help > 0 && !u.classes.is_empty() {
            let best = regions
                .iter()
                .filter(|(_, (at, pr, ps))| {
                    at.first == layout.heads
                        && at.roots > 0
                        && pr.tables.is_empty()
                        && pr.all_reduces == 0
                        && !pr.csl.contains("rp.tables(")
                        && ps.iter().all(|x| *x != ParamAt::Resident)
                })
                .filter(|(i, _)| {
                    matches!(
                        u.steps[**i],
                        Step::Region {
                            side: Side::Col,
                            ..
                        }
                    )
                })
                .max_by(|a, b| {
                    let c = |(at, pr, _): &(place::RegionAt, Printed, Vec<ParamAt>)| {
                        crate::cost::stream::RegionFeatures::of(&pr.csl, at.chunk).cycles(at.chunk)
                    };
                    c(a.1).total_cmp(&c(b.1))
                })
                .map(|(i, (at, _, ps))| (*i, *at, ps.clone()));
            if let Some((i, at, ps)) = best {
                // the most PEs (root and helpers) up to help + 1 that cut the chunk evenly
                let d = (1..=split.help + 1)
                    .rev()
                    .find(|d| at.chunk.is_multiple_of(*d) && at.chunk / d >= 4)
                    .unwrap_or(1);
                if d > 1
                    && let Step::Region { chain, .. } = &u.steps[i]
                {
                    let top = class_rows.iter().map(|c| c.0).max().unwrap_or(1).max(2) - 1;
                    let lv = (32 - top.leading_zeros()).div_ceil(3);
                    let pr = region::print(
                        &chain.region,
                        &Place {
                            chunk: at.chunk / d,
                            roots: at.roots * d,
                            params: ps,
                            round16,
                            rope_lv: lv.max(1),
                            rope_bits: 3,
                            straddle: false,
                        },
                    )
                    .map_err(|e| format!("{}: step {i} (helpers): {e}", u.name))?;
                    helper = Some((i, d - 1, pr));
                }
            }
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
            helper,
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

    // the floorplan: each block's K / V columns its own attention's (its first unit's), the
    // band as tall as the tallest
    let joint_of = |k: &KvSched| {
        k.units
            .first()
            .and_then(|&u| built.get(u))
            .and_then(|b| b.unit.joint)
    };
    let heads = kv
        .iter()
        .filter_map(|k| joint_of(k).map(|j| j.kv_heads))
        .max()
        .unwrap_or(shape.attn.kv_heads);
    let requests = Requests {
        stream: true,
        joint: !blocks.is_empty(),
        lanes: lane_unit,
        ..Requests::default()
    };
    let pya = kv
        .iter()
        .map(|k| {
            // a recurrence's column: its state's rows over the fewest PEs (a decode step's a
            // state a lane slot, a row a lane)
            if joint_of(k).is_some_and(|j| j.state.is_some()) {
                let lanes = shape
                    .cache
                    .map_or(1, |_| class_rows.iter().map(|c| c.0).sum::<u32>());
                return PYA * lanes.max(1);
            }
            let (qh, hd) = joint_of(k).map_or(
                (
                    shape.attn.q_heads / shape.attn.kv_heads.max(1),
                    shape.attn.head_dim,
                ),
                |j| (j.q_heads / j.kv_heads.max(1), j.head_dim),
            );
            kv_pya_split(
                shape
                    .cache
                    .unwrap_or_else(|| k.stores.iter().map(|s| s.1).sum()),
                qh,
                hd,
                split,
            )
        })
        .max()
        .unwrap_or(PYA);
    // helpers below a unit's rest roots: within the band's K / V rows (where a K / V column
    // takes several columns, above the K / V roots' row, which they push down), and not under a
    // block whose later units' links pass west below its first unit to K / V columns there (at
    // `sub > 1` they pass above them)
    let space = if blocks.is_empty() {
        0
    } else if split.sub > 1 {
        u32::MAX
    } else {
        heads + pya.div_ceil(split.sub.max(1))
    };
    for (bi, b) in built.iter_mut().enumerate() {
        let under = split.sub == 1 && kv.iter().any(|k| k.units.len() > 1 && k.units[0] == bi);
        if under || b.helper.as_ref().is_some_and(|h| h.1 >= space) {
            b.helper = None;
        }
    }
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
            helpers: b.helper.as_ref().map_or(0, |h| h.1),
        })
        .collect();
    let stream = Stream::of(&requests, py, &asks, &blocks, heads, pya, split.sub, bounds)?;
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
    use crate::floor::Coord;

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

    /// flux2-mini within a fabric 38 PEs wide (as `wafer-compile stream --fabric-width 38`):
    /// three bands, a later band's first unit under the lane unit, and the lane bus reaching
    /// every root that reads a lane vector in each of them.
    #[test]
    fn flux_2_mini_over_three_bands_reaches_every_lane_root() {
        let t = trace();
        let s = super::super::shape_of(&t, "denoise", 16).unwrap_or_else(|e| panic!("{e}"));
        let p = program_py(
            &t,
            s,
            true,
            None,
            &[],
            None,
            None,
            None,
            false,
            Bounds::system().narrowed(Some(38)),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let st = &p.stream;
        let bands: std::collections::BTreeSet<u32> = st.units.iter().map(|u| u.y).collect();
        assert_eq!(bands.len(), 3, "{} x {}", st.w, st.h);
        assert!(st.w <= 38);
        assert!(st.units.iter().any(|u| u.x == 0 && u.y > 0 && u.on_path));
        let lb = st.lanes.as_ref().expect("a lane bus");
        let w = st.wiring();
        let bus = w
            .links
            .iter()
            .find_map(|l| match l {
                crate::wiring::Link::Broadcast(b) => Some(b),
                _ => None,
            })
            .expect("the lane bus");
        for c in &lb.to {
            assert!(
                bus.pts
                    .iter()
                    .position(|q| q == c)
                    .is_some_and(|i| bus.recv[i]),
                "no lane bus at {c:?}"
            );
        }
        // every weight PE's casts in (its unit's row and column casts) bound to a colour routed
        // to its ramp there: a later band's weight PEs share their columns with the first band's
        let e = crate::emit::stream::emit(&p).unwrap_or_else(|e| panic!("{e}"));
        let layout = &e
            .files
            .iter()
            .find(|(n, _)| n == "layout.csl")
            .expect("layout.csl")
            .1;
        let at = |l: &str, k: &str| -> Option<u32> {
            let i = l.find(k)? + k.len();
            l[i..]
                .split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse()
                .ok()
        };
        let xy = |l: &str| -> Option<(u32, u32)> {
            let a = l.split_once('(')?.1;
            let mut it = a.split(',');
            Some((
                it.next()?.trim().parse().ok()?,
                it.next()?.trim().parse().ok()?,
            ))
        };
        let to_ramp: std::collections::BTreeSet<(u32, u32, u32)> = layout
            .lines()
            .filter(|l| l.contains("@set_color_config(") && l.contains("RAMP } }"))
            .filter_map(|l| Some((xy(l)?, at(l, "@get_color(")?)))
            .map(|((x, y), c)| (x, y, c))
            .collect();
        let mut n = 0;
        for l in layout
            .lines()
            .filter(|l| l.contains("@set_tile_code(") && l.contains("\"w_"))
        {
            let (x, y) = xy(l).expect("a PE");
            for k in [".C_XR = ", ".C_XC = "] {
                let c = at(l, k).expect("a cast colour");
                assert!(
                    to_ramp.contains(&(x, y, c)),
                    "({x}, {y}){k}{c}: not to its ramp"
                );
            }
            n += 1;
        }
        assert!(n > 0);
    }

    /// The row store: flux2-mini's attending units with their held rows in their weight PEs
    /// (each row root's chunk cast to its row, summed back up the row chain after the barrier)
    /// build and print: the CX roots hold one row of each, the weight PEs their share.
    #[test]
    fn the_row_store_holds_a_units_rows_in_its_weight_pes() {
        let t = trace();
        let s = super::super::shape_of(&t, "denoise", 64).unwrap_or_else(|e| panic!("{e}"));
        let p = program_py(
            &t,
            s,
            true,
            None,
            &[],
            Some(16),
            Some(place::TILE_BUDGET),
            None,
            true,
            Bounds::system(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let spilled: Vec<&str> = p
            .units
            .iter()
            .filter(|b| b.layout.spill)
            .map(|b| b.unit.name.as_str())
            .collect();
        assert_eq!(spilled.len(), 6, "{spilled:?}");
        let e = crate::emit::stream::emit(&p).unwrap_or_else(|e| panic!("{e}"));
        let file = |n: &str| &e.files.iter().find(|(f, _)| f == n).expect(n).1;
        assert!(file("w_3_0.csl").contains("const HSW: u16 = 16;"));
        assert!(file("cx_3_0.csl").contains("fab.fin(q_ci, 16)"));
    }

    /// The PEs of one kind share their images: a unit's CX roots a few files, a K / V column's PEs
    /// one file and colours, their differences (lane drains, a PE's index in its column) the
    /// config words the host writes; identical kernels across units one file.
    #[test]
    fn pes_of_a_kind_share_an_image() {
        let t = trace();
        let s = super::super::shape_of(&t, "denoise", 64).unwrap_or_else(|e| panic!("{e}"));
        let p = program(&t, s, true).unwrap_or_else(|e| panic!("{e}"));
        let e = crate::emit::stream::emit(&p).unwrap_or_else(|e| panic!("{e}"));
        let layout = &e
            .files
            .iter()
            .find(|(f, _)| f == "layout.csl")
            .expect("layout")
            .1;
        let code = |c: Coord| -> &str {
            let pre = format!("@set_tile_code({}, {}, \"", c.x, c.y);
            let at = layout.find(&pre).expect("a PE's code") + pre.len();
            &layout[at..at + layout[at..].find('"').expect("a file")]
        };
        // (a unit's lane words may fall on its roots' chunks unevenly: a few files then)
        for su in &p.stream.units {
            let files: std::collections::BTreeSet<&str> =
                (0..su.rows()).map(|j| code(su.cx(j))).collect();
            assert!(files.len() <= 4, "{}: {files:?}", su.name);
        }
        // (file, params): an image; the memcpy params' column aside
        let images: std::collections::BTreeSet<(&str, &str)> = layout
            .lines()
            .filter(|l| l.contains("@set_tile_code("))
            .filter_map(|l| {
                let (_, rest) = l.split_once(", \"")?;
                let (file, rest) = rest.split_once('"')?;
                Some((file, rest.split_once(')').map_or(rest, |x| x.1)))
            })
            .collect();
        let files = e.files.iter().filter(|(f, _)| f != "layout.csl").count();
        eprintln!(
            "{} PEs, {} images, {files} files",
            p.stream.w * p.stream.h,
            images.len()
        );
        // (a folded K / V column's first unit takes a few more: its pitched columns' chains)
        assert!(images.len() < 140, "{} images", images.len());
        let pcfg = e.arrays.iter().filter(|a| a.sym == "pcfg").count();
        assert!(pcfg > 0);
    }
}
