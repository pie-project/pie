//! A stream reading's wafer program, end to end before the CSL: its units (the lane prologue's,
//! then a unit a side of every stage in path order: [`super::unit`]), their layouts and the row
//! stream's schedule ([`super::place`]), the floorplan's elements ([`crate::compile::floorplan::stream`])
//! and the fabric they are laid on (coloured by [`Palette::allocate`]).

use std::collections::{BTreeMap, BTreeSet};

use poem_compiler::spatial::layer::stream::{Stage, StreamProgram};
use poem_ir::{Def, Linear, Operands, Operation, Trace, ValueId};

use super::StreamShape;
use super::place::{self, KvSched, Layout, UnitSched};
use super::unit::{self, Reader, Side, Step, Unit};
use crate::compile::emit::region::{self, ParamAt, Place, Printed};
use crate::compile::fabric::{Fabric, Palette};
use crate::compile::floorplan::Requests;
use crate::compile::floorplan::stream::{Bounds, Stream, UnitAsk};

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

/// The attention PEs a K / V column of `rows` stored rows takes: a column as long as its entries
/// ask, row `p` kept by PE `p % pya`.
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

/// The most replicas of a K / V head the search tries ([`program_at`]'s `rep`).
pub const MAX_KV_REP: u32 = 16;

/// `p`'s PEs' memory, the static check's model of it as emitted ([`crate::compile::check::PeBytes`]):
/// each kind of PE's estimate leaves the stack's room ([`crate::compile::build::STACK_ROOM`]); past it,
/// the next cheapest candidate.
fn fits(p: &Program) -> R<()> {
    let e = crate::compile::emit::stream::emit(p)?;
    let rep = crate::compile::check::placed(&e.placed, &e.files, None);
    if let Some((f, d, c)) = rep.codes.iter().find(|(_, data, code)| {
        data + code + i64::from(crate::compile::build::STACK_ROOM) > crate::compile::check::PE_BYTES
    }) {
        return Err(format!(
            "{}: `{f}`: {d} B data + ~{c} B code past a PE's {} B less the stack's {} B",
            p.trace.name,
            crate::compile::check::PE_BYTES,
            crate::compile::build::STACK_ROOM
        ));
    }
    Ok(())
}

/// The cheapest candidates a replica count builds that the next count prices again.
const KV_REP_TRIES: usize = 4;

/// The cuts the search tries: the fewest rest roots, a quarter and a half more of them; helpers
/// below the rest roots; K / V columns over `sub` columns a head at `1 / sub` of the entries a
/// PE (as many rows of the band, `sub` times the PEs: a query's work over them), the head
/// columns of a block's first unit `sub` apart ([`crate::compile::floorplan::stream::KvColumn`]).
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
/// the barrier in the row store always), or (`None`) at the (row roots, tile budget) the cost
/// model prices lowest of those that build within `bounds`.
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
    // pinned too)
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
    // every candidate built, priced: (cycles, candidate, replica count)
    let mut priced: Vec<(f64, (u32, u32, KvSplit), u32)> = Vec::new();
    let mut errs: Vec<(u32, u32, String)> = Vec::new();
    // each K / V head over 1, 2, 4 .. replicas (a head root's q heads and tiles fewer, more
    // K / V PEs at once), each count priced like any other choice: every candidate at one
    // replica, the next count over the cheapest few that built (every candidate while none has),
    // up to the count no attention takes
    let mut rep = 1;
    let mut tries = cands.clone();
    while rep <= MAX_KV_REP && !tries.is_empty() {
        let mut built: Vec<(f64, (u32, u32, KvSplit))> = Vec::new();
        let mut applied = false;
        for group in tries.chunks(8) {
            let tried: Vec<((u32, u32, KvSplit), Made)> = std::thread::scope(|sc| {
                let hs: Vec<_> = group
                    .iter()
                    .map(|&(p, tile, kv)| {
                        let shape = shape.clone();
                        sc.spawn(move || {
                            let r = program_at(
                                trace, shape, round16, upto, taps, p, tile, kv, spill, bounds, rep,
                            );
                            ((p, tile, kv), r)
                        })
                    })
                    .collect();
                group
                    .iter()
                    .zip(hs)
                    .map(|(&c, h)| {
                        h.join()
                            .unwrap_or_else(|_| (c, Err(Unbuilt::Refused("panicked".into()))))
                    })
                    .collect()
            });
            for (cand, r) in tried {
                match r {
                    Ok(prog) => {
                        applied = true;
                        let c = crate::compile::cost::stream::time(&prog).cycles;
                        built.push((c, cand));
                        priced.push((c, cand, rep));
                        if best.as_ref().is_none_or(|(b, _)| c < *b) {
                            best = Some((c, prog));
                        }
                    }
                    // (a replica count no attention takes says nothing of the candidate)
                    Err(Unbuilt::NoReplicas(_)) => {}
                    Err(Unbuilt::Refused(e)) => {
                        applied = true;
                        errs.push((cand.0, cand.1, e));
                    }
                }
            }
        }
        if !applied {
            break;
        }
        tries = if built.is_empty() {
            cands.clone()
        } else {
            built.sort_by(|a, b| a.0.total_cmp(&b.0));
            built.iter().take(KV_REP_TRIES).map(|&(_, c)| c).collect()
        };
        rep *= 2;
    }
    // the cheapest held to the PEs' memory (the static check's data of it as emitted), else
    // the next cheapest, rebuilt
    if let Some((c0, p0)) = best.take() {
        priced.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut first = Some(p0);
        for &(c, (p, tile, kv), rep) in &priced {
            let prog = match first.take().filter(|_| c == c0) {
                Some(prog) => prog,
                None => match program_at(
                    trace,
                    shape.clone(),
                    round16,
                    upto,
                    taps,
                    p,
                    tile,
                    kv,
                    spill,
                    bounds,
                    rep,
                ) {
                    Ok(prog) => prog,
                    Err(_) => continue,
                },
            };
            match fits(&prog) {
                Ok(()) => {
                    best = Some((c, prog));
                    break;
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

/// Why [`program_at`] builds no program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unbuilt {
    /// No attention's q heads split over this many K / V replicas: the count, not the
    /// candidate, is what does not apply.
    NoReplicas(u32),
    /// The candidate does not build, and why.
    Refused(String),
}

/// What [`program_at`] makes of a candidate.
type Made = Result<Program, Unbuilt>;

impl From<String> for Unbuilt {
    fn from(why: String) -> Unbuilt {
        Unbuilt::Refused(why)
    }
}

impl std::fmt::Display for Unbuilt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unbuilt::NoReplicas(rep) => write!(f, "no attention's q heads split {rep} ways"),
            Unbuilt::Refused(why) => f.write_str(why),
        }
    }
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
    rep: u32,
) -> Made {
    let prog = &shape.program;
    let name = &trace.name;
    if py == 0 || !prog.hidden.is_multiple_of(py) {
        return Err(Unbuilt::Refused(format!(
            "{name}: a hidden width of {} cuts over no {py} rows",
            prog.hidden
        )));
    }
    let Lowered {
        class_rows,
        mut units,
        blocks,
        ..
    } = lower_units(trace, prog, upto, taps)?;
    // each attention's K / V heads replicated `rep` times, a replica its share of the q heads
    // (a head root's tiles that many times fewer)
    if rep > 1 {
        let mut any = false;
        for (u, _) in &mut units {
            if let Some(j) = u.joint.as_mut()
                && crate::kernel::column(Some(j)).is_ok_and(|c| c.replicates())
                && (j.q_heads / j.kv_heads.max(1)).is_multiple_of(rep)
            {
                j.kv_heads *= rep;
                j.kv_rep = rep;
                any = true;
            }
        }
        if !any {
            return Err(Unbuilt::NoReplicas(rep));
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
        // (a row-side value its row roots hold padded, narrower than them or of no multiple: a
        // row region runs over its padded chunks, its outputs' padding zeroed)
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
            // positions below a class's rows (a decode step's: below its cache's positions)
            let (lv, bits) = rope_digits(&class_rows, shape.cache);
            let place = Place {
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
                rope_bits: bits,
                straddle: false,
                columns: None,
                dup: at.per.max(1),
            };
            // a root kernel's step: its module, not the region printer's
            let printed = match crate::kernel::root_of(trace, chain) {
                Some(k) => k.print(trace, chain, &place, u.rows, shape.cache.is_some()),
                None => region::print(&chain.region, &place).map_err(|e| e.to_string()),
            }
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
                        &u.steps[**i],
                        Step::Region {
                            side: Side::Col,
                            chain,
                        } if crate::kernel::root_of(trace, chain).is_none()
                    )
                })
                .max_by(|a, b| {
                    let c = |(at, pr, _): &(place::RegionAt, Printed, Vec<ParamAt>)| {
                        crate::compile::cost::stream::RegionFeatures::of(&pr.csl, at.chunk)
                            .cycles(at.chunk)
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
                    let (lv, bits) = rope_digits(&class_rows, shape.cache);
                    let pr = region::print(
                        &chain.region,
                        &Place {
                            chunk: at.chunk / d,
                            roots: at.roots * d,
                            params: ps,
                            round16,
                            rope_lv: lv.max(1),
                            rope_bits: bits,
                            straddle: false,
                            columns: None,
                            dup: 1,
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
    // (a unit takes its rows from the host where it reads the host's values and no unit before
    // it on the path runs its classes; a later one's host values join the rows the stream
    // brings it: a per-layer embedding's lookup)
    let path_units: Vec<(Vec<usize>, bool, bool, bool)> = path
        .iter()
        .enumerate()
        .map(|(pi, &i)| {
            let u = &built[i].unit;
            let first = !path[..pi]
                .iter()
                .any(|&j| built[j].unit.classes.iter().any(|c| u.classes.contains(c)));
            (
                u.classes.clone(),
                first
                    && u.steps
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
            // each column's PEs its mixer's
            let j = joint_of(k);
            let (q_per_kv, head_dim) = j.map_or(
                (
                    shape.attn.q_heads / shape.attn.kv_heads.max(1),
                    shape.attn.head_dim,
                ),
                |j| (j.q_heads / j.kv_heads.max(1), j.head_dim),
            );
            let ask = crate::kernel::ColumnAsk {
                q_per_kv,
                head_dim,
                positions: shape
                    .cache
                    .unwrap_or_else(|| k.stores.iter().map(|s| s.1).sum()),
                lanes: shape
                    .cache
                    .map_or(1, |_| class_rows.iter().map(|c| c.0).sum::<u32>()),
                split,
            };
            crate::kernel::column(j.as_ref()).map(|c| c.pya(&ask))
        })
        .collect::<R<Vec<u32>>>()?
        .into_iter()
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
        .map(|b| -> R<UnitAsk> {
            Ok(UnitAsk {
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
                col_segment: col_segment(b)?,
                on_path: !b.unit.classes.is_empty(),
                lanes: b.lanes,
                helpers: b.helper.as_ref().map_or(0, |h| h.1),
            })
        })
        .collect::<R<_>>()?;
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

/// A rotation table's digits (levels, bits a level) for the positions it serves: a class's
/// rows in digits of 3 bits; a decode step's (a row at any position below its cache's) none,
/// its angles computed at the root from the frequencies alone (`srope`'s DIRECT, BITS 0).
fn rope_digits(class_rows: &[(u32, u32)], cache: Option<u32>) -> (u32, u32) {
    if cache.is_some() {
        return (1, 0);
    }
    let rows = class_rows.iter().map(|c| c.0).max().unwrap_or(1);
    let top = rows.max(2) - 1;
    ((32 - top.leading_zeros()).div_ceil(3).max(1), 3)
}

/// The CY roots a unit's column regions all-reduce over (`(first, roots)`: one segment
/// collective, so every column region that sums across roots spans the same roots), none where
/// no column region sums across roots.
fn col_segment(b: &Built) -> R<Option<(u32, u32)>> {
    let spans: BTreeSet<(u32, u32)> = b
        .regions
        .iter()
        .filter(|(si, (at, p, _))| {
            at.roots > 1
                && p.all_reduces > 0
                && matches!(
                    b.unit.steps[**si],
                    Step::Region {
                        side: Side::Col,
                        ..
                    }
                )
        })
        .map(|(_, (at, _, _))| (at.first, at.roots))
        .collect();
    match spans.len() {
        0 => Ok(None),
        1 => Ok(spans.into_iter().next()),
        _ => Err(format!(
            "{}: column regions all-reduce over different roots ({spans:?}): one segment a unit's column roots",
            b.unit.name
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::floor::Coord;

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

    /// The PEs of one kind share their images: a unit's CX roots a few files, a K / V column's PEs
    /// one file and colours, their differences (lane drains, a PE's index in its column) the
    /// config words the host writes; identical kernels across units one file.
    #[test]
    fn pes_of_a_kind_share_an_image() {
        let t = trace();
        let s = super::super::shape_of(&t, "denoise", 64).unwrap_or_else(|e| panic!("{e}"));
        let p = program(&t, s, true).unwrap_or_else(|e| panic!("{e}"));
        let e = crate::compile::emit::stream::emit(&p).unwrap_or_else(|e| panic!("{e}"));
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
