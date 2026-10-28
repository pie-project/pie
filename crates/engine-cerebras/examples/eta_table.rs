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

use engine_cerebras::spatial::guest3::*;

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
        let seeds: BTreeMap<u32, Value> = case.seeds.iter().cloned().collect();
        let record = match translate(&case.container, &seeds, vocab) {
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
