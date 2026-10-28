//! eta guest programs as op tables for the wafer's guest region.
//!
//! The guest region (docs/cerebras-spatial-design.md §4a, the `guest3`
//! interpreter under spatial/bench-job/payload) runs a program that is data:
//! five-word instructions over wide registers (one a vocabulary row, spread
//! over a lane's PEs) and scalar registers. This example builds the sampling
//! epilogues the inferlets use, translates each to such a table, and writes
//! test cases whose golden outputs come from eta's host interpreter
//! (`eta_exec::step`) on the same inputs.
//!
//! The random stream's `[key, counter]` lives in the state scalar registers
//! (S0..S3, written by the host per request, kept across steps): the keyed
//! draws run on the PEs (`WRNG`: eta-ir's SplitMix64 in software u64, bit for
//! bit; the Gumbel transform through a musl-style `logf`), the `+ iota(2)`
//! is a u32 add, and a stream put back with the input's shape is carried into
//! the state registers for the next step. Constant U32 values stay on the host.
//!
//! ```text
//! cargo run --release -p engine-cerebras --example eta_table -- <out-dir> [vocab]
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use eta_compiler::plan::compile_bound;
use eta_exec::{ExecPlan, PassInputs, StepOutcome, Value};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Literal, RngKind, Shape};
use serde_json::{Value as Json, json};

// The guest3 opcodes (payload/guest3/pe.csl).
const WADD: i32 = 2;
const WSUBS: i32 = 3;
const WMULS: i32 = 4;
const WEXP: i32 = 5;
const RMAX: i32 = 6;
const RSUM: i32 = 7;
const RARGMAX: i32 = 8;
const OUT: i32 = 10;
const WDIVS: i32 = 11;
const WMUL: i32 = 13;
const WSUB: i32 = 14;
const WADDS: i32 = 15;
const WMAX: i32 = 16;
const WMIN: i32 = 17;
const WGT: i32 = 18;
const WGTS: i32 = 19;
const WSEL: i32 = 20;
const WFILLS: i32 = 22;
const SCONST: i32 = 23;
const SDIV: i32 = 24;
const SMUL: i32 = 25;
const SADD: i32 = 26;
const SSUB: i32 = 27;
const WNEG: i32 = 28;
const WLT: i32 = 29;
const WLTS: i32 = 30;
const RMIN: i32 = 32;
const WEQS: i32 = 33;
const WRNG: i32 = 34;
const WLOG: i32 = 35;
const SIADD: i32 = 36;
const SMOV: i32 = 37;

const NREG: u32 = 12;
const NSREG: u32 = 24;
/// Scalar registers S0..NSTATE hold the program's state (the random stream).
const NSTATE: u32 = 4;

// ------------------------------------------------------------------ programs

struct Program {
    ops: Vec<Op>,
    next: u32,
    channels: Vec<ChannelDecl>,
}

impl Program {
    fn new() -> Program {
        Program {
            ops: Vec::new(),
            next: 0,
            channels: Vec::new(),
        }
    }
    fn op(&mut self, op: Op) -> u32 {
        let id = self.next;
        self.next += op.result_count();
        self.ops.push(op);
        id
    }
    fn input(&mut self, shape: Shape, dtype: Dtype) -> (u32, u32) {
        let chan = self.channels.len() as u32;
        self.channels.push(ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::Writer,
            seeded: true,
        });
        (chan, self.op(Op::ChanTake(chan)))
    }
    fn output(&mut self, shape: Shape, dtype: Dtype, value: u32) -> u32 {
        let chan = self.channels.len() as u32;
        self.channels.push(ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        });
        self.op(Op::ChanPut { chan, value });
        chan
    }
    fn f(&mut self, x: f32) -> u32 {
        self.op(Op::Const(Literal::F32(x)))
    }
    fn logits(&mut self, vocab: u32) -> u32 {
        let raw = self.op(Op::Output {
            index: 0,
            lane: 0,
            shape: Shape::matrix(1, vocab),
            dtype: Dtype::F32,
        });
        self.op(Op::Reshape {
            value: raw,
            shape: Shape::vector(vocab),
        })
    }
    fn container(self) -> TraceContainer {
        TraceContainer {
            names: vec!["logits".to_string()],
            channels: self.channels,
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: self.ops,
            }],
            externs: Vec::new(),
            inputs: Vec::new(),
        }
    }
}

/// A test program: its container, its output channels, its seeded inputs.
struct Case {
    name: &'static str,
    container: TraceContainer,
    outputs: Vec<u32>,
    seeds: Vec<(u32, Value)>,
}

/// Greedy decoding: the argmax of the logits.
fn greedy(vocab: u32) -> Case {
    let mut p = Program::new();
    let logits = p.logits(vocab);
    let token = p.op(Op::ReduceArgmax(logits));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let out = p.output(Shape::vector(1), Dtype::I32, token1);
    Case {
        name: "greedy",
        container: p.container(),
        outputs: vec![out],
        seeds: Vec::new(),
    }
}

/// naive-baseline's epilogue: logits / T, a Gumbel-max draw, the stream on.
fn naive(vocab: u32, temperature: f32, name: &'static str) -> Case {
    let mut p = Program::new();
    let (c_rng, r) = p.input(Shape::vector(2), Dtype::U32);
    let logits = p.logits(vocab);
    let scaled = if temperature == 1.0 {
        logits
    } else {
        let t = p.f(temperature);
        p.op(Op::Div(logits, t))
    };
    let noise = p.op(Op::RngKeyed {
        state: r,
        shape: Shape::vector(vocab),
        kind: RngKind::Gumbel,
    });
    let perturbed = p.op(Op::Add(scaled, noise));
    let token = p.op(Op::ReduceArgmax(perturbed));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let step = p.op(Op::Iota { len: 2 });
    let r_next = p.op(Op::Add(r, step));
    let outs = vec![
        p.output(Shape::vector(1), Dtype::I32, token1),
        p.output(Shape::vector(2), Dtype::U32, r_next),
    ];
    Case {
        name,
        container: p.container(),
        outputs: outs,
        seeds: vec![(c_rng, Value::U32(vec![0x7ce1, 3]))],
    }
}

/// Softmax at a temperature, with its statistics and the greedy token.
fn softmax_stats(vocab: u32) -> Case {
    let v = Shape::vector(vocab);
    let mut p = Program::new();
    let logits = p.logits(vocab);
    let t = p.f(0.7);
    let scaled = p.op(Op::Div(logits, t));
    let m = p.op(Op::ReduceMax(scaled));
    let mb = p.op(Op::Broadcast { value: m, shape: v });
    let c = p.op(Op::Sub(scaled, mb));
    let e = p.op(Op::Exp(c));
    let s = p.op(Op::ReduceSum(e));
    let sb = p.op(Op::Broadcast { value: s, shape: v });
    let probs = p.op(Op::Div(e, sb));
    let token = p.op(Op::ReduceArgmax(probs));
    let lo = p.op(Op::ReduceMin(scaled));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let m1 = p.op(Op::Reshape {
        value: m,
        shape: Shape::vector(1),
    });
    let s1 = p.op(Op::Reshape {
        value: s,
        shape: Shape::vector(1),
    });
    let lo1 = p.op(Op::Reshape {
        value: lo,
        shape: Shape::vector(1),
    });
    let outs = vec![
        p.output(Shape::vector(1), Dtype::I32, token1),
        p.output(Shape::vector(1), Dtype::F32, m1),
        p.output(Shape::vector(1), Dtype::F32, s1),
        p.output(Shape::vector(1), Dtype::F32, lo1),
        p.output(v, Dtype::F32, probs),
    ];
    Case {
        name: "softmax_stats",
        container: p.container(),
        outputs: outs,
        seeds: Vec::new(),
    }
}

/// Repetition, frequency and presence penalties (the penalty inferlets'
/// epilogue without top-k/top-p), then temperature and a Gumbel-max draw.
fn penalties(vocab: u32) -> Case {
    let v = Shape::vector(vocab);
    let mut p = Program::new();
    let (c_rng, r) = p.input(Shape::vector(2), Dtype::U32);
    let (c_counts, counts) = p.input(v, Dtype::F32);
    let (c_present, present) = p.input(v, Dtype::F32);
    let logits = p.logits(vocab);
    let zero = p.f(0.0);
    let zb = p.op(Op::Broadcast {
        value: zero,
        shape: v,
    });
    let seen_out = p.op(Op::Gt(counts, zb));
    let half = p.f(0.5);
    let hb = p.op(Op::Broadcast {
        value: half,
        shape: v,
    });
    let seen_prompt = p.op(Op::Gt(present, hb));
    let seen = p.op(Op::Or(seen_out, seen_prompt));
    let rp = p.f(1.3);
    let rpb = p.op(Op::Broadcast {
        value: rp,
        shape: v,
    });
    let positive = p.op(Op::Gt(logits, zb));
    let shrunk = p.op(Op::Div(logits, rpb));
    let grown = p.op(Op::Mul(logits, rpb));
    let rep = p.op(Op::Select {
        cond: positive,
        a: shrunk,
        b: grown,
    });
    let logits = p.op(Op::Select {
        cond: seen,
        a: rep,
        b: logits,
    });
    let fp = p.f(0.2);
    let fpb = p.op(Op::Broadcast {
        value: fp,
        shape: v,
    });
    let freq = p.op(Op::Mul(fpb, counts));
    let logits = p.op(Op::Sub(logits, freq));
    let pp = p.f(0.4);
    let ppb = p.op(Op::Broadcast {
        value: pp,
        shape: v,
    });
    let so = p.op(Op::Cast {
        value: seen_out,
        dtype: Dtype::F32,
    });
    let pres = p.op(Op::Mul(ppb, so));
    let logits = p.op(Op::Sub(logits, pres));
    let temp = p.f(0.7);
    let scaled = p.op(Op::Div(logits, temp));
    let drawn = p.op(Op::RngKeyed {
        state: r,
        shape: v,
        kind: RngKind::Gumbel,
    });
    let perturbed = p.op(Op::Add(scaled, drawn));
    let token = p.op(Op::ReduceArgmax(perturbed));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let step = p.op(Op::Iota { len: 2 });
    let r_next = p.op(Op::Add(r, step));
    let outs = vec![
        p.output(Shape::vector(1), Dtype::I32, token1),
        p.output(Shape::vector(2), Dtype::U32, r_next),
        p.output(v, Dtype::F32, scaled),
    ];
    let n = vocab as usize;
    let mut cnt = vec![0.0f32; n];
    let mut pre = vec![0.0f32; n];
    for i in 0..n {
        if (i * 7 + 3) % 11 == 0 {
            cnt[i] = 1.0 + (i % 3) as f32;
        }
        if (i * 5 + 1) % 13 == 0 {
            pre[i] = 1.0;
        }
    }
    Case {
        name: "penalties",
        container: p.container(),
        outputs: outs,
        seeds: vec![
            (c_rng, Value::U32(vec![0x51a7, 9])),
            (c_counts, Value::F32(cnt)),
            (c_present, Value::F32(pre)),
        ],
    }
}

/// The full sampler with top-k/top-p cuts and a grammar mask: kept to show
/// what falls back to the host (PivotThreshold, MaskApply, ScatterAdd).
fn top_k_top_p(vocab: u32) -> Case {
    let v = Shape::vector(vocab);
    let mut p = Program::new();
    let (c_rng, r) = p.input(Shape::vector(2), Dtype::U32);
    let logits = p.logits(vocab);
    let temp = p.f(0.7);
    let scaled = p.op(Op::Div(logits, temp));
    let m = p.op(Op::ReduceMax(scaled));
    let mb = p.op(Op::Broadcast { value: m, shape: v });
    let c = p.op(Op::Sub(scaled, mb));
    let e = p.op(Op::Exp(c));
    let s = p.op(Op::ReduceSum(e));
    let sb = p.op(Op::Broadcast { value: s, shape: v });
    let probs = p.op(Op::Div(e, sb));
    let k = p.op(Op::Const(Literal::U32(12)));
    let keep = p.op(Op::PivotThreshold {
        input: probs,
        predicate: eta_ir::types::Predicate::RankLe(k),
    });
    let ninf = p.f(f32::NEG_INFINITY);
    let cut = p.op(Op::Select {
        cond: keep,
        a: scaled,
        b: ninf,
    });
    let drawn = p.op(Op::RngKeyed {
        state: r,
        shape: v,
        kind: RngKind::Gumbel,
    });
    let perturbed = p.op(Op::Add(cut, drawn));
    let token = p.op(Op::ReduceArgmax(perturbed));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let outs = vec![p.output(Shape::vector(1), Dtype::I32, token1)];
    Case {
        name: "top_k",
        container: p.container(),
        outputs: outs,
        seeds: vec![(c_rng, Value::U32(vec![0x7ce1, 4]))],
    }
}

// ------------------------------------------------------------------ the host interpreter

fn bind_logits(
    container: TraceContainer,
    vocab: u32,
) -> Result<eta_ir::validate::BoundTrace, eta_ir::validate::ValidateError> {
    eta_ir::validate::bind(
        container,
        vec![vec![eta_ir::registry::OutputDecl {
            name: "logits".to_string(),
            axis: eta_ir::registry::Axis::Row,
            width: vocab,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }]],
        eta_ir::registry::EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
}

fn plan_of(container: TraceContainer, vocab: u32) -> ExecPlan {
    let bound = bind_logits(container, vocab).expect("the program binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let plan = eta_exec::adopt_launch_package(launch).expect("the package adopts");
    assert!(plan.executable, "{:?}", plan.reject_reason);
    plan
}

/// eta_exec's answer on every output channel.
fn golden(case: &Case, vocab: u32, logits: &[f32]) -> Vec<Value> {
    step_golden(case, vocab, logits, case.seeds.iter().cloned().collect())
}

/// `steps` steps on the same logits, each step's random stream the one the
/// step before put back (as a decode loop feeds it).
fn sequence(case: &Case, vocab: u32, logits: &[f32], steps: usize) -> Vec<Vec<Value>> {
    let mut seeds: BTreeMap<u32, Value> = case.seeds.iter().cloned().collect();
    let Some((&chan, Value::U32(st))) = seeds.iter().find(|(_, v)| matches!(v, Value::U32(_)))
    else {
        return Vec::new();
    };
    let len = st.len();
    (0..steps)
        .map(|_| {
            let outs = step_golden(case, vocab, logits, seeds.clone());
            let next = outs
                .iter()
                .find(|v| matches!(v, Value::U32(x) if x.len() == len))
                .expect("the stream is put back")
                .clone();
            seeds.insert(chan, next);
            outs
        })
        .collect()
}

fn step_golden(case: &Case, vocab: u32, logits: &[f32], seeds: BTreeMap<u32, Value>) -> Vec<Value> {
    let plan = plan_of(case.container.clone(), vocab);
    let mut inst = eta_exec::make_host_instance(&plan, &BTreeMap::new(), &seeds);
    let outputs = vec![("logits".to_string(), 0, Value::F32(logits.to_vec()))];
    let outcome = eta_exec::step(&mut inst, &plan, &PassInputs { outputs: &outputs });
    assert_eq!(
        outcome,
        StepOutcome::Committed,
        "{}: the interpreter commits",
        case.name
    );
    case.outputs
        .iter()
        .map(|&c| match eta_exec::host_take(&inst, &plan, c) {
            (eta_exec::HostOp::Ok, Some(v)) => v,
            other => panic!("{}: channel {c} holds nothing: {other:?}", case.name),
        })
        .collect()
}

fn numel_of(v: &Value) -> usize {
    match v {
        Value::F32(x) => x.len(),
        Value::I32(x) => x.len(),
        Value::U32(x) => x.len(),
        Value::Bool(x) => x.len(),
    }
}

fn json_of(v: &Value) -> Json {
    match v {
        Value::F32(x) => json!(
            x.iter()
                .map(|f| if f.is_finite() {
                    json!(f)
                } else {
                    json!(f.to_string())
                })
                .collect::<Vec<_>>()
        ),
        Value::I32(x) => json!(x),
        Value::U32(x) => json!(x),
        Value::Bool(x) => json!(x),
    }
}

// ------------------------------------------------------------------ translation

/// Where an eta value lives once translated.
#[derive(Clone, Debug)]
enum Loc {
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

struct Table {
    ins: Vec<[i32; 5]>,
    wide_inputs: Vec<(u32, Vec<f32>)>,
    outputs: Vec<Json>,
    nreg: u32,
    nsreg: u32,
    /// Wide registers free for reuse (their values are past their last use).
    free: Vec<u32>,
    /// The state registers' initial values.
    state: Vec<u32>,
    /// State registers a step's put carries into the next step: (to, from).
    carry: Vec<(u32, u32)>,
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

fn numel(shape: &Shape) -> u32 {
    shape.dims().iter().product::<u32>().max(1)
}

/// One step's table: register 0 holds the logits; channel inputs the host
/// seeded are loaded as wide registers or kept on the host.
fn translate(case: &Case, vocab: u32) -> Result<Table, String> {
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
    let ops_all = &case.container.stages[0].ops;
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
    let seeds: BTreeMap<u32, Value> = case.seeds.iter().cloned().collect();
    let chans = &case.container.channels;
    let ops = &case.container.stages[0].ops;
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
            Op::Broadcast { value, .. } => match get(*value)? {
                Loc::Scalar(s) => Some(Loc::Bcast(s)),
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

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args.next().expect("usage: eta_table <out-dir> [vocab]");
    let vocab: u32 = args
        .next()
        .map_or(250, |v| v.parse().expect("a vocabulary size"));
    std::fs::create_dir_all(&out_dir).expect("the output directory");
    let cases = vec![
        greedy(vocab),
        naive(vocab, 0.7, "naive_t0.7"),
        naive(vocab, 1.0, "naive_t1"),
        softmax_stats(vocab),
        penalties(vocab),
        top_k_top_p(vocab),
    ];
    // A logits plane with ties and a range a sampler sees.
    let logits: Vec<f32> = (0..vocab)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(0x9E37_79B9) % 10007) as f32 / 10007.0;
            (x - 0.5) * 12.0 + if i % 97 == 5 { 3.0 } else { 0.0 }
        })
        .collect();
    let mut summary = Vec::new();
    for case in &cases {
        let gold = golden(case, vocab, &logits);
        let record = match translate(case, vocab) {
            Ok(table) => {
                // The host-kept outputs must already agree with eta.
                for (o, g) in table.outputs.iter().zip(&gold) {
                    if o["kind"] == "host" {
                        assert_eq!(
                            o["value"],
                            json_of(g),
                            "{}: a host-kept output differs",
                            case.name
                        );
                    }
                }
                let outputs: Vec<Json> = table
                    .outputs
                    .iter()
                    .zip(&gold)
                    .map(|(o, g)| {
                        let mut o = o.clone();
                        o["golden"] = json_of(g);
                        o
                    })
                    .collect();
                json!({
                    "name": case.name, "vocab": vocab, "translated": true,
                    "nreg": table.nreg, "nsreg": table.nsreg,
                    "table": table.ins,
                    "logits": logits,
                    "wide_inputs": table.wide_inputs.iter().map(|(r, x)| json!({"reg": r, "values": x})).collect::<Vec<_>>(),
                    "outputs": outputs,
                    "state": table.state,
                    "sequence": sequence(case, vocab, &logits, if table.state.is_empty() { 0 } else { 64 })
                        .iter()
                        .map(|outs| outs.iter()
                            // Only the small outputs: a step's tokens and stream.
                            .map(|v| if numel_of(v) <= 16 { json_of(v) } else { Json::Null }).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                })
            }
            Err(why) => json!({"name": case.name, "vocab": vocab, "translated": false, "why": why}),
        };
        eprintln!(
            "{:<14} {}",
            case.name,
            if record["translated"] == true {
                format!(
                    "{} instructions, {} wide registers",
                    record["table"].as_array().map_or(0, Vec::len),
                    record["nreg"]
                )
            } else {
                format!("falls back to the host: {}", record["why"])
            }
        );
        summary.push(json!({"name": case.name, "translated": record["translated"], "why": record.get("why")}));
        let path = Path::new(&out_dir).join(format!("{}.json", case.name));
        std::fs::write(&path, serde_json::to_string(&record).expect("json")).expect("a case file");
    }
    std::fs::write(
        Path::new(&out_dir).join("summary.json"),
        serde_json::to_string_pretty(&summary).expect("json"),
    )
    .expect("the summary");
}
