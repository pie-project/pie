//! (The eta_table example's guest3 translator; the engine uses guest4.)
//!
//! eta epilogues as op tables for the wafer's guest region (the `guest3`
//! interpreter, docs/cerebras-spatial-design.md §4a; spatial/eta/README.md in
//! the bring-up tree): five-word instructions over wide registers (one a
//! vocabulary row over a lane's PEs) and scalar registers. Register 0 holds
//! the logits; host-seeded channel inputs become wide registers the host
//! loads or, for the random stream, state scalar registers S0..S3 the host
//! writes per request and the table carries from step to step.

use std::collections::BTreeMap;

use eta_exec::Value;
use eta_ir::container::TraceContainer;
use eta_ir::op::Op;
use eta_ir::types::{Literal, RngKind, Shape};
use serde_json::{Value as Json, json};

// The guest3 opcodes (payload/guest3/pe.csl).
pub const WADD: i32 = 2;
pub const WSUBS: i32 = 3;
pub const WMULS: i32 = 4;
pub const WEXP: i32 = 5;
pub const RMAX: i32 = 6;
pub const RSUM: i32 = 7;
pub const RARGMAX: i32 = 8;
pub const OUT: i32 = 10;
pub const WDIVS: i32 = 11;
pub const WMUL: i32 = 13;
pub const WSUB: i32 = 14;
pub const WADDS: i32 = 15;
pub const WMAX: i32 = 16;
pub const WMIN: i32 = 17;
pub const WGT: i32 = 18;
pub const WGTS: i32 = 19;
pub const WSEL: i32 = 20;
pub const WFILLS: i32 = 22;
pub const SCONST: i32 = 23;
pub const SDIV: i32 = 24;
pub const SMUL: i32 = 25;
pub const SADD: i32 = 26;
pub const SSUB: i32 = 27;
pub const WNEG: i32 = 28;
pub const WLT: i32 = 29;
pub const WLTS: i32 = 30;
pub const RMIN: i32 = 32;
pub const WEQS: i32 = 33;
pub const WRNG: i32 = 34;
pub const WLOG: i32 = 35;
pub const SIADD: i32 = 36;
pub const SMOV: i32 = 37;

pub const NREG: u32 = 12;
pub const NSREG: u32 = 24;
/// Scalar registers S0..NSTATE hold the program's state (the random stream).
pub const NSTATE: u32 = 4;

// ------------------------------------------------------------------ translation

/// Where an eta value lives once translated.
#[derive(Clone, Debug)]
pub enum Loc {
    Wide(u32),
    Scalar(u32),
    /// A scalar broadcast to the vocabulary (not materialized).
    Bcast(u32),
    /// U32 values in scalar registers, as their bits (the random stream).
    State(Vec<u32>),
    /// A small integer constant the host keeps (an iota).
    Host(Vec<u32>),
    /// A U32 constant the host keeps (a top-k k).
    HostConst,
}

pub struct Table {
    pub ins: Vec<[i32; 5]>,
    pub wide_inputs: Vec<(u32, Vec<f32>)>,
    pub outputs: Vec<Json>,
    pub nreg: u32,
    pub nsreg: u32,
    /// Wide registers free for reuse (their values are past their last use).
    free: Vec<u32>,
    /// The state registers' initial values.
    pub state: Vec<u32>,
    /// State registers a step's put carries into the next step: (to, from).
    pub carry: Vec<(u32, u32)>,
}

impl Table {
    fn emit(&mut self, op: i32, d: u32, a: u32, b: u32, imm: i32) {
        self.ins.push([op, d as i32, a as i32, b as i32, imm]);
    }
    fn wide(&mut self) -> Result<u32, String> {
        if let Some(r) = self.free.pop() {
            return Ok(r);
        }
        self.fresh()
    }
    /// A register no instruction has written: the host loads inputs into
    /// registers before the table runs, so an input never takes a reused one.
    fn fresh(&mut self) -> Result<u32, String> {
        if self.nreg >= NREG {
            return Err(format!("more than {NREG} wide values"));
        }
        self.nreg += 1;
        Ok(self.nreg - 1)
    }
    fn scalar(&mut self) -> Result<u32, String> {
        if self.nsreg >= NSREG {
            return Err(format!("more than {NSREG} scalar values"));
        }
        self.nsreg += 1;
        Ok(self.nsreg - 1)
    }
    fn sconst(&mut self, x: f32) -> Result<u32, String> {
        let s = self.scalar()?;
        self.emit(SCONST, s, 0, 0, x.to_bits() as i32);
        Ok(s)
    }
    /// A wide register holding `l` (a broadcast scalar is filled in).
    fn as_wide(&mut self, l: &Loc) -> Result<u32, String> {
        match l {
            Loc::Wide(r) => Ok(*r),
            Loc::Bcast(s) | Loc::Scalar(s) => {
                let r = self.wide()?;
                self.emit(WFILLS, r, *s, 0, 0);
                Ok(r)
            }
            other => Err(format!("{other:?} is not a device value")),
        }
    }
}

pub fn numel(shape: &Shape) -> u32 {
    shape.dims().iter().product::<u32>().max(1)
}

/// One step's table: register 0 holds the logits; channel inputs the host
/// seeded are loaded as wide registers or kept on the host.
pub fn translate(
    container: &TraceContainer,
    seeds: &BTreeMap<u32, Value>,
    vocab: u32,
) -> Result<Table, String> {
    let mut t = Table {
        ins: Vec::new(),
        wide_inputs: Vec::new(),
        outputs: Vec::new(),
        nreg: 1,
        nsreg: NSTATE,
        free: Vec::new(),
        state: Vec::new(),
        carry: Vec::new(),
    };
    // Each value's last reader (by op index), and the values a wide output
    // keeps alive to the end: a wide register is reused once every value
    // living in it is past its last reader.
    let stage = container
        .stages
        .iter()
        .find(|s| s.stage == eta_ir::registry::Stage::Epilogue)
        .ok_or("no epilogue stage")?;
    let ops_all = &stage.ops;
    let mut last_use: BTreeMap<u32, usize> = BTreeMap::new();
    let mut keep: Vec<u32> = Vec::new();
    for (i, op) in ops_all.iter().enumerate() {
        for v in operands(op) {
            last_use.insert(v, i);
        }
        if let Op::ChanPut { value, .. } = op {
            keep.push(*value);
        }
    }
    let mut loc: BTreeMap<u32, Loc> = BTreeMap::new();
    let chans = &container.channels;
    let ops = &stage.ops;
    let mut id = 0u32;
    let mut out_slot = 0u32;
    for (index, op) in ops.iter().enumerate() {
        let here = id;
        id += op.result_count();
        let get = |v: u32| {
            loc.get(&v)
                .cloned()
                .ok_or_else(|| format!("value {v} untranslated"))
        };
        let result: Option<Loc> = match op {
            Op::Output { .. } => Some(Loc::Wide(0)),
            Op::Reshape { value, .. } => Some(get(*value)?),
            Op::Cast { value, .. } => Some(get(*value)?),
            Op::Const(Literal::F32(x)) => Some(Loc::Scalar(t.sconst(*x)?)),
            Op::Const(Literal::U32(_)) => Some(Loc::HostConst),
            Op::Const(other) => return Err(format!("unsupported constant {other:?}")),
            Op::ChanTake(c) => {
                let decl = &chans[*c as usize];
                let n = numel(&decl.shape);
                match seeds.get(c) {
                    Some(Value::F32(x)) if n == vocab => {
                        let r = t.fresh()?;
                        t.wide_inputs.push((r, x.clone()));
                        Some(Loc::Wide(r))
                    }
                    Some(Value::Bool(x)) if n == vocab => {
                        let r = t.fresh()?;
                        t.wide_inputs
                            .push((r, x.iter().map(|&b| b as f32).collect()));
                        Some(Loc::Wide(r))
                    }
                    Some(Value::U32(x)) if t.state.is_empty() && n <= NSTATE => {
                        t.state = x.clone();
                        Some(Loc::State((0..n).collect()))
                    }
                    other => return Err(format!("unsupported channel input {c}: {other:?}")),
                }
            }
            Op::Broadcast { value, shape } => match get(*value)? {
                Loc::Scalar(s) => Some(Loc::Bcast(s)),
                // A vocabulary row broadcast to a vocabulary row's shape
                // ([V] to [1, V]) is the row.
                Loc::Wide(r) if numel(shape) == vocab => Some(Loc::Wide(r)),
                other => return Err(format!("broadcast of {other:?}")),
            },
            Op::Iota { len } => Some(Loc::Host((0..*len).collect())),
            Op::RngKeyed { state, shape, kind } => match get(*state)? {
                Loc::State(st) if numel(shape) == vocab => {
                    let gumbel = match kind {
                        RngKind::Uniform => 0,
                        RngKind::Gumbel => 1,
                        RngKind::Normal => return Err("normal draws".into()),
                    };
                    let ctr = match st.get(1) {
                        Some(&c) => c,
                        None => t.sconst(0.0)?,
                    };
                    let d = t.wide()?;
                    t.emit(WRNG, d, st[0], ctr, gumbel);
                    Some(Loc::Wide(d))
                }
                other => return Err(format!("keyed draws from {other:?}")),
            },
            Op::Log(a) => {
                let ra = t.as_wide(&get(*a)?)?;
                let d = t.wide()?;
                t.emit(WLOG, d, ra, 0, 0);
                Some(Loc::Wide(d))
            }
            Op::Exp(a) => {
                let ra = t.as_wide(&get(*a)?)?;
                let d = t.wide()?;
                t.emit(WEXP, d, ra, 0, 0);
                Some(Loc::Wide(d))
            }
            Op::Neg(a) => {
                let ra = t.as_wide(&get(*a)?)?;
                let d = t.wide()?;
                t.emit(WNEG, d, ra, 0, 0);
                Some(Loc::Wide(d))
            }
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => {
                let (la, lb) = (get(*a)?, get(*b)?);
                let kind = match op {
                    Op::Add(..) => 0,
                    Op::Sub(..) => 1,
                    Op::Mul(..) => 2,
                    _ => 3,
                };
                Some(binary(&mut t, kind, la, lb)?)
            }
            Op::MaxElem(a, b) | Op::Or(a, b) => {
                let (ra, rb) = (t.as_wide(&get(*a)?)?, t.as_wide(&get(*b)?)?);
                let d = t.wide()?;
                t.emit(WMAX, d, ra, rb, 0);
                Some(Loc::Wide(d))
            }
            Op::MinElem(a, b) | Op::And(a, b) => {
                let (ra, rb) = (t.as_wide(&get(*a)?)?, t.as_wide(&get(*b)?)?);
                let d = t.wide()?;
                t.emit(WMIN, d, ra, rb, 0);
                Some(Loc::Wide(d))
            }
            // a >= b is 1 - (a < b), a <= b is 1 - (a > b) (0/1 rows).
            Op::Ge(a, b) | Op::Le(a, b) => {
                let ge = matches!(op, Op::Ge(..));
                let (la, lb) = (get(*a)?, get(*b)?);
                let lt = t.wide()?;
                match (&la, &lb) {
                    (Loc::Wide(ra), Loc::Wide(rb)) => {
                        t.emit(if ge { WLT } else { WGT }, lt, *ra, *rb, 0)
                    }
                    (Loc::Wide(ra), Loc::Bcast(s) | Loc::Scalar(s)) => {
                        t.emit(if ge { WLTS } else { WGTS }, lt, *ra, *s, 0)
                    }
                    (Loc::Bcast(s) | Loc::Scalar(s), Loc::Wide(rb)) => {
                        t.emit(if ge { WGTS } else { WLTS }, lt, *rb, *s, 0)
                    }
                    other => return Err(format!("comparison of {other:?}")),
                }
                let one = t.sconst(1.0)?;
                let neg = t.wide()?;
                t.emit(WNEG, neg, lt, 0, 0);
                t.free.push(lt);
                let d = t.wide()?;
                t.emit(WADDS, d, neg, one, 0);
                t.free.push(neg);
                Some(Loc::Wide(d))
            }
            Op::Gt(a, b) | Op::Lt(a, b) => {
                let gt = matches!(op, Op::Gt(..));
                let (la, lb) = (get(*a)?, get(*b)?);
                let d = t.wide()?;
                match (&la, &lb) {
                    (Loc::Wide(ra), Loc::Wide(rb)) => {
                        t.emit(if gt { WGT } else { WLT }, d, *ra, *rb, 0)
                    }
                    (Loc::Wide(ra), Loc::Bcast(s) | Loc::Scalar(s)) => {
                        t.emit(if gt { WGTS } else { WLTS }, d, *ra, *s, 0)
                    }
                    (Loc::Bcast(s) | Loc::Scalar(s), Loc::Wide(rb)) => {
                        t.emit(if gt { WLTS } else { WGTS }, d, *rb, *s, 0)
                    }
                    other => return Err(format!("comparison of {other:?}")),
                }
                Some(Loc::Wide(d))
            }
            Op::Eq(a, b) => match (get(*a)?, get(*b)?) {
                (Loc::Wide(ra), Loc::Bcast(s) | Loc::Scalar(s)) => {
                    let d = t.wide()?;
                    t.emit(WEQS, d, ra, s, 0);
                    Some(Loc::Wide(d))
                }
                other => return Err(format!("eq of {other:?}")),
            },
            Op::Select { cond, a, b } => {
                let rc = t.as_wide(&get(*cond)?)?;
                let (ra, rb) = (t.as_wide(&get(*a)?)?, t.as_wide(&get(*b)?)?);
                let d = t.wide()?;
                t.emit(WSEL, d, ra, rb, rc as i32);
                Some(Loc::Wide(d))
            }
            Op::ReduceMax(a) | Op::ReduceSum(a) | Op::ReduceMin(a) | Op::ReduceArgmax(a) => {
                let ra = t.as_wide(&get(*a)?)?;
                let s = t.scalar()?;
                let code = match op {
                    Op::ReduceMax(..) => RMAX,
                    Op::ReduceSum(..) => RSUM,
                    Op::ReduceMin(..) => RMIN,
                    _ => RARGMAX,
                };
                t.emit(code, s, ra, 0, 0);
                Some(Loc::Scalar(s))
            }
            Op::ChanPut { chan, value } => {
                let entry = match get(*value)? {
                    Loc::Scalar(s) => {
                        let slot = out_slot;
                        out_slot += 1;
                        if slot >= 8 {
                            return Err("more than 8 scalar outputs".into());
                        }
                        t.emit(OUT, slot, s, 0, 0);
                        json!({"chan": chan, "kind": "scalar", "slot": slot})
                    }
                    Loc::Wide(r) => json!({"chan": chan, "kind": "wide", "reg": r}),
                    Loc::State(st) => {
                        let slots: Vec<u32> = (out_slot..out_slot + st.len() as u32).collect();
                        out_slot += st.len() as u32;
                        if out_slot > 8 {
                            return Err("more than 8 scalar outputs".into());
                        }
                        for (&slot, &s) in slots.iter().zip(&st) {
                            t.emit(OUT, slot, s, 0, 0);
                        }
                        // The stream put back carries into the next step.
                        if st.len() == t.state.len() && t.carry.is_empty() {
                            t.carry = st.iter().enumerate().map(|(i, &s)| (i as u32, s)).collect();
                        }
                        json!({"chan": chan, "kind": "state", "slots": slots})
                    }
                    Loc::Host(x) => json!({"chan": chan, "kind": "host", "value": x}),
                    other => return Err(format!("put of {other:?}")),
                };
                t.outputs.push(entry);
                None
            }
            other => {
                return Err(format!(
                    "unsupported op {}",
                    format!("{other:?}")
                        .split(['(', ' ', '{'])
                        .next()
                        .unwrap_or("?")
                ));
            }
        };
        if let Some(l) = result {
            loc.insert(here, l);
        }
        // Free the wide registers whose every value died at this op.
        let dead: Vec<u32> = (operands(op).into_iter())
            .filter(|v| last_use.get(v) == Some(&index))
            .collect();
        for v in dead {
            let Some(Loc::Wide(r)) = loc.get(&v).cloned() else {
                continue;
            };
            // Host-loaded inputs stay: the next step (the host does not
            // reload them) reads them again.
            if r == 0 || t.free.contains(&r) || t.wide_inputs.iter().any(|(w, _)| *w == r) {
                continue;
            }
            let alive = loc.iter().any(|(&w, l)| {
                matches!(l, Loc::Wide(x) if *x == r)
                    && (keep.contains(&w) || last_use.get(&w).is_some_and(|&u| u > index))
            });
            if !alive {
                t.free.push(r);
            }
        }
    }
    let carry = t.carry.clone();
    for (to, from) in carry {
        if to != from {
            t.emit(SMOV, to, from, 0, 0);
        }
    }
    Ok(t)
}

/// The values an op reads (the variants the translator handles).
fn operands(op: &Op) -> Vec<u32> {
    match op {
        Op::Reshape { value, .. } | Op::Cast { value, .. } | Op::Broadcast { value, .. } => {
            vec![*value]
        }
        Op::Exp(a)
        | Op::Log(a)
        | Op::Neg(a)
        | Op::ReduceMax(a)
        | Op::ReduceSum(a)
        | Op::ReduceMin(a)
        | Op::ReduceArgmax(a) => vec![*a],
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::MaxElem(a, b)
        | Op::MinElem(a, b)
        | Op::Or(a, b)
        | Op::And(a, b)
        | Op::Gt(a, b)
        | Op::Lt(a, b)
        | Op::Ge(a, b)
        | Op::Le(a, b)
        | Op::Eq(a, b) => vec![*a, *b],
        Op::Select { cond, a, b } => vec![*cond, *a, *b],
        Op::RngKeyed { state, .. } => vec![*state],
        Op::ChanPut { value, .. } => vec![*value],
        Op::PivotThreshold { input, .. } => vec![*input],
        _ => Vec::new(),
    }
}

/// Add / Sub / Mul / Div (kind 0..3) of two translated values.
fn binary(t: &mut Table, kind: u8, la: Loc, lb: Loc) -> Result<Loc, String> {
    use Loc::*;
    Ok(match (la, lb) {
        (State(x), Host(y)) if kind == 0 => {
            let mut out = Vec::new();
            for (&a, &b) in x.iter().zip(y.iter().cycle()) {
                let s = t.scalar()?;
                t.emit(SIADD, s, a, 0, b as i32);
                out.push(s);
            }
            State(out)
        }
        (Host(x), Host(y)) if kind == 0 => Host(
            x.iter()
                .zip(y.iter().cycle())
                .map(|(a, b)| a.wrapping_add(*b))
                .collect(),
        ),
        (Scalar(a), Scalar(b)) => {
            let s = t.scalar()?;
            t.emit([SADD, SSUB, SMUL, SDIV][kind as usize], s, a, b, 0);
            Scalar(s)
        }
        (Wide(a), Wide(b)) => {
            if kind == 3 {
                return Err("wide / wide (no vector divide)".into());
            }
            let d = t.wide()?;
            t.emit([WADD, WSUB, WMUL, 0][kind as usize], d, a, b, 0);
            Wide(d)
        }
        (Wide(a), Bcast(s) | Scalar(s)) => {
            let d = t.wide()?;
            t.emit([WADDS, WSUBS, WMULS, WDIVS][kind as usize], d, a, s, 0);
            Wide(d)
        }
        (Bcast(s) | Scalar(s), Wide(b)) => match kind {
            0 | 2 => {
                let d = t.wide()?;
                t.emit(if kind == 0 { WADDS } else { WMULS }, d, b, s, 0);
                Wide(d)
            }
            1 => {
                // s - w = -w + s
                let n = t.wide()?;
                t.emit(WNEG, n, b, 0, 0);
                let d = t.wide()?;
                t.emit(WADDS, d, n, s, 0);
                Wide(d)
            }
            _ => return Err("scalar / wide".into()),
        },
        (a, b) => return Err(format!("arithmetic on {a:?}, {b:?}")),
    })
}
