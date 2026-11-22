//! Every guest stage the device runs answers what the host interpreter
//! (`eta_exec::step`) answers on the same inputs: bit for bit where the
//! interpreter is exact (integers, orderings and their ties, masks, argmax,
//! the uniform draw), within rounding for float math.
//!
//! Each program is stepped twice from the same seeds: once by the
//! interpreter over a host logits plane, once with every stage lowered to
//! StableHLO and run on the device against the same plane uploaded as the
//! fire's readout. A group of instances of one program also steps as one
//! batched executable and must answer, lane by lane, what each instance
//! answers alone.

use std::collections::BTreeMap;

use engine_xla::device::Device;
use engine_xla::guest::run::{Bound, Member, OnDevice, Stages, step_group};
use engine_xla::readout::{Kept, Plane, Seat};
use eta_compiler::plan::compile_bound;
use eta_exec::{ExecPlan, InterpInstance, PassInputs, StepOutcome, Value};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, Stage};
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape};

// ------------------------------------------------------------------ builder

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

fn plan_of(container: TraceContainer, vocab: u32) -> ExecPlan {
    plan_over(container, vocab, 1)
}

/// The program bound over `lanes` lanes, each returning its logits.
fn plan_over(container: TraceContainer, vocab: u32, lanes: usize) -> ExecPlan {
    let logits = eta_ir::registry::OutputDecl {
        name: "logits".to_string(),
        axis: eta_ir::registry::Axis::Row,
        width: vocab,
        dtype: eta_ir::types::Dtype::F32,
        layers: 0,
        readout: true,
    };
    let bound = eta_ir::validate::bind(
        container,
        vec![vec![logits]; lanes],
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the program binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let plan = eta_exec::adopt_launch_package(launch).expect("the package adopts");
    assert!(plan.executable, "{:?}", plan.reject_reason);
    if let Err(why) = engine_xla::guest::admits(&plan.package) {
        panic!("the program has no device form: {why}");
    }
    plan
}

fn key_of(plan: &ExecPlan) -> [u8; 32] {
    *blake3::hash(format!("{:?}", plan.package).as_bytes()).as_bytes()
}

fn instance(plan: &ExecPlan, seeds: &[(u32, Value)]) -> InterpInstance {
    let seeds: BTreeMap<u32, Value> = seeds.iter().cloned().collect();
    eta_exec::make_host_instance(plan, &BTreeMap::new(), &seeds)
}

fn take_all(plan: &ExecPlan, inst: &InterpInstance, channels: &[u32]) -> Vec<Value> {
    channels
        .iter()
        .map(|&c| match eta_exec::host_take(inst, plan, c) {
            (eta_exec::HostOp::Ok, Some(v)) => v,
            other => panic!("channel {c} holds nothing: {other:?}"),
        })
        .collect()
}

/// The interpreter's answer for one lane reading `logits` (`rows` rows).
fn host(
    plan: &ExecPlan,
    seeds: &[(u32, Value)],
    logits: &[f32],
    _rows: u32,
    channels: &[u32],
) -> Vec<Value> {
    let mut inst = instance(plan, seeds);
    let outputs: Vec<(String, u32, Value)> = if logits.is_empty() {
        Vec::new()
    } else {
        vec![("logits".to_string(), 0, Value::F32(logits.to_vec()))]
    };
    let inputs = PassInputs { outputs: &outputs };
    let outcome = eta_exec::step(&mut inst, plan, &inputs);
    assert_eq!(outcome, StepOutcome::Committed, "the interpreter commits");
    take_all(plan, &inst, channels)
}

struct Dev {
    device: Device,
    _lock: engine_xla::bench::DeviceLock,
}

fn device() -> Option<Dev> {
    if std::env::var_os("PIE_XLA_PLUGIN").is_none()
        && std::env::var_os("TPU_LIBRARY_PATH").is_none()
    {
        eprintln!("no PJRT plugin named: skipping");
        return None;
    }
    let lock = engine_xla::bench::lock_device();
    match Device::open(None, 0) {
        Ok(device) => Some(Dev {
            device,
            _lock: lock,
        }),
        Err(e) => {
            eprintln!("no device ({e}): skipping");
            None
        }
    }
}

fn kept(device: &Device, plane: &[f32], rows: u32, layout: Vec<(u32, u32)>) -> Kept {
    let width = plane.len() as u32 / rows.max(1);
    let bytes: Vec<u8> = plane.iter().flat_map(|x| x.to_le_bytes()).collect();
    let buffer = device
        .upload(Dtype::F32, rows, width, &bytes)
        .expect("the plane uploads");
    let plane = Plane::new(
        "logits".to_string(),
        poem_ir::InputAxis::Row,
        true,
        std::sync::Arc::new(buffer),
        rows,
        width,
        0,
        Dtype::F32,
    );
    Kept {
        planes: vec![plane],
        seats: (layout.iter())
            .map(|&(first, count)| vec![Some(Seat { first, count })])
            .collect(),
        readback: vec![None; layout.len()],
        pixels: (String::new(), Vec::new()),
        host: Vec::new(),
    }
}

/// Where the program reads its outputs for the fire's lane `lane`,
/// attached over that one lane.
fn bound(plan: &ExecPlan, kept: &Kept, lane: usize) -> Vec<Bound> {
    engine_xla::guest::run::bound(plan, Some(kept), lane as u32, 1)
        .expect("the program reads its own lane")
        .expect("the lane binds")
}

/// The device's answer for one lane reading its outputs where `bound` says.
fn on_device(
    device: &Device,
    stages: &mut Stages,
    plan: &ExecPlan,
    seeds: &[(u32, Value)],
    kept: Option<&Kept>,
    bound: Vec<Bound>,
    channels: &[u32],
) -> Vec<Value> {
    let mut inst = instance(plan, seeds);
    let mut runner = OnDevice::new(device, stages, key_of(plan), kept, bound);
    let outcome = eta_exec::step_with(
        &mut inst,
        plan,
        &PassInputs::none(),
        &mut runner,
        eta_ir::registry::Stage::Epilogue,
    );
    assert_eq!(outcome, StepOutcome::Committed, "the device pass commits");
    assert!(runner.ran() > 0, "a stage ran on the device");
    take_all(plan, &inst, channels)
}

fn close(got: &Value, want: &Value, tol: f32, what: &str) {
    match (got, want) {
        (Value::F32(g), Value::F32(w)) => {
            assert_eq!(g.len(), w.len(), "{what}: lengths");
            for (i, (x, y)) in g.iter().zip(w).enumerate() {
                let same = (x.is_nan() && y.is_nan()) || x == y;
                assert!(
                    same || (x - y).abs() <= tol * (1.0 + y.abs()),
                    "{what}[{i}]: device {x} vs interpreter {y}"
                );
            }
        }
        _ => panic!("{what}: {got:?} vs {want:?} are not both f32"),
    }
}

fn exact(got: &Value, want: &Value, what: &str) {
    if let (Value::F32(g), Value::F32(w)) = (got, want) {
        let gb: Vec<u32> = g.iter().map(|x| x.to_bits()).collect();
        let wb: Vec<u32> = w.iter().map(|x| x.to_bits()).collect();
        assert_eq!(gb, wb, "{what}: bits");
        return;
    }
    assert_eq!(got, want, "{what}");
}

fn noise(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
            lo + (hi - lo) * u
        })
        .collect()
}

fn logits_plane(rows: u32, vocab: u32, seed: u64) -> Vec<f32> {
    let mut plane = noise(seed, (rows * vocab) as usize, -4.0, 4.0);
    for row in 0..rows as usize {
        for c in 0..vocab as usize {
            plane[row * vocab as usize + c] += (c % 7) as f32 * 0.01;
        }
        plane[row * vocab as usize + (row * 97 + 5) % vocab as usize] = 9.0 + row as f32;
    }
    plane
}

// --------------------------------------------------------------- programs

const ROWS: u32 = 8;
const VOCAB: u32 = 4096;

/// engine-cuda's epilogue: softmax, gather_row, argmax, top-k, row sums.
fn statistics() -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let (_, temp) = p.input(Shape::vector(1), Dtype::F32);
    let temp = p.op(Op::Reshape {
        value: temp,
        shape: Shape::new(&[]).unwrap(),
    });
    let (_, index) = p.input(Shape::vector(ROWS), Dtype::I32);
    let logits = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(ROWS, VOCAB),
        dtype: Dtype::F32,
    });
    let scaled = p.op(Op::Div(logits, temp));
    let peak = p.op(Op::ReduceMax(scaled));
    let peak = p.op(Op::Reshape {
        value: peak,
        shape: Shape::matrix(ROWS, 1),
    });
    let peak = p.op(Op::Broadcast {
        value: peak,
        shape: Shape::matrix(ROWS, VOCAB),
    });
    let centered = p.op(Op::Sub(scaled, peak));
    let e = p.op(Op::Exp(centered));
    let s = p.op(Op::ReduceSum(e));
    let s = p.op(Op::Reshape {
        value: s,
        shape: Shape::matrix(ROWS, 1),
    });
    let s = p.op(Op::Broadcast {
        value: s,
        shape: Shape::matrix(ROWS, VOCAB),
    });
    let probs = p.op(Op::Div(e, s));
    let gathered = p.op(Op::GatherRow {
        src: scaled,
        idx: index,
    });
    let argmax = p.op(Op::ReduceArgmax(scaled));
    let top = p.op(Op::TopK {
        input: scaled,
        k: 4,
    });
    let mass = p.op(Op::ReduceSum(probs));
    let arg_u = p.op(Op::Cast {
        value: argmax,
        dtype: Dtype::U32,
    });
    let at_peak = p.op(Op::GatherRow {
        src: probs,
        idx: arg_u,
    });
    let outs = vec![
        p.output(Shape::vector(ROWS), Dtype::F32, gathered),
        p.output(Shape::vector(ROWS), Dtype::I32, argmax),
        p.output(Shape::matrix(ROWS, 4), Dtype::U32, top + 1),
        p.output(Shape::matrix(ROWS, 4), Dtype::F32, top),
        p.output(Shape::vector(ROWS), Dtype::F32, mass),
        p.output(Shape::vector(ROWS), Dtype::F32, at_peak),
    ];
    (p.container(), outs)
}

/// A beam's pick: the top-k over every row of the logits as one vector (a
/// beam inferlet's top-b of b x vocab scores), at three widths.
fn flat_top_k() -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let logits = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(ROWS, VOCAB),
        dtype: Dtype::F32,
    });
    let flat = p.op(Op::Reshape {
        value: logits,
        shape: Shape::vector(ROWS * VOCAB),
    });
    let four = p.op(Op::TopK { input: flat, k: 4 });
    let nine = p.op(Op::TopK { input: flat, k: 9 });
    let one = p.op(Op::TopK { input: flat, k: 1 });
    let outs = vec![
        p.output(Shape::vector(4), Dtype::U32, four + 1),
        p.output(Shape::vector(4), Dtype::F32, four),
        p.output(Shape::vector(9), Dtype::U32, nine + 1),
        p.output(Shape::vector(9), Dtype::F32, nine),
        p.output(Shape::vector(1), Dtype::U32, one + 1),
        p.output(Shape::vector(1), Dtype::F32, one),
    ];
    (p.container(), outs)
}

/// A Gumbel draw fused (`gumbel_argmax`) and unfused (a multiply by the
/// scale, `select`, `add` of `rng_keyed`'s noise, `reduce_argmax`) over f32
/// logits, and fused over their bf16 rounding; `keep` masks the lanes at or
/// under `floor`.
fn gumbel_draws() -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let (_, scale) = p.input(Shape::vector(1), Dtype::F32);
    let scale = p.op(Op::Reshape {
        value: scale,
        shape: Shape::new(&[]).unwrap(),
    });
    let (_, state) = p.input(Shape::vector(2), Dtype::U32);
    let logits = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(ROWS, VOCAB),
        dtype: Dtype::F32,
    });
    let floor = p.f(-3.0);
    let keep = p.op(Op::Gt(logits, floor));
    let fused = p.op(Op::GumbelArgmax {
        logits,
        keep,
        scale,
        state,
    });
    let scaled = p.op(Op::Mul(logits, scale));
    let ninf = p.f(f32::NEG_INFINITY);
    let masked = p.op(Op::Select {
        cond: keep,
        a: scaled,
        b: ninf,
    });
    let noise = p.op(Op::RngKeyed {
        state,
        shape: Shape::matrix(ROWS, VOCAB),
        kind: RngKind::Gumbel,
    });
    let perturbed = p.op(Op::Add(masked, noise));
    let unfused = p.op(Op::ReduceArgmax(perturbed));
    let half = p.op(Op::Cast {
        value: logits,
        dtype: Dtype::Bf16,
    });
    let fused_half = p.op(Op::GumbelArgmax {
        logits: half,
        keep,
        scale,
        state,
    });
    let outs = vec![
        p.output(Shape::vector(ROWS), Dtype::I32, fused),
        p.output(Shape::vector(ROWS), Dtype::I32, unfused),
        p.output(Shape::vector(ROWS), Dtype::I32, fused_half),
    ];
    (p.container(), outs)
}

/// The bf16 rules over a bf16 rounding of the logits: an elementwise op's
/// bf16 result (rounded once), a row sum (f32), a scan (bf16, each prefix
/// rounded once), a top-k (the input's values, exact, and their ids).
fn bf16_rules() -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let logits = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(ROWS, VOCAB),
        dtype: Dtype::F32,
    });
    let half = p.op(Op::Cast {
        value: logits,
        dtype: Dtype::Bf16,
    });
    let third = p.f(0.333_333_34);
    let scaled = p.op(Op::Mul(half, third));
    let sum = p.op(Op::ReduceSum(half));
    let scan = p.op(Op::CumSum(scaled));
    let top = p.op(Op::TopK { input: half, k: 4 });
    let outs = vec![
        p.output(Shape::matrix(ROWS, VOCAB), Dtype::Bf16, scaled),
        p.output(Shape::vector(ROWS), Dtype::F32, sum),
        p.output(Shape::matrix(ROWS, VOCAB), Dtype::Bf16, scan),
        p.output(Shape::matrix(ROWS, 4), Dtype::U32, top + 1),
        p.output(Shape::matrix(ROWS, 4), Dtype::Bf16, top),
    ];
    (p.container(), outs)
}

/// The logits cast to bf16 and put out as they are: the cast's rounding
/// alone.
fn bf16_cast() -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let logits = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(ROWS, VOCAB),
        dtype: Dtype::F32,
    });
    let half = p.op(Op::Cast {
        value: logits,
        dtype: Dtype::Bf16,
    });
    let outs = vec![p.output(Shape::matrix(ROWS, VOCAB), Dtype::Bf16, half)];
    (p.container(), outs)
}

/// The compat sampler over one row: penalties over a carried histogram, a
/// grammar mask (bools and packed words), temperature, top-k, top-p and a
/// keyed Gumbel draw; plus the greedy token and the next histogram.
fn sampler(vocab: u32) -> (TraceContainer, Vec<u32>, [u32; 5]) {
    let v = Shape::vector(vocab);
    let mut p = Program::new();
    let (c_rng, r) = p.input(Shape::vector(2), Dtype::U32);
    let (c_counts, counts) = p.input(v, Dtype::F32);
    let (c_present, present) = p.input(v, Dtype::F32);
    let (c_mask, mask) = p.input(v, Dtype::Bool);
    let (c_words, words) = p.input(Shape::vector(vocab.div_ceil(32)), Dtype::U32);
    let raw = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, vocab),
        dtype: Dtype::F32,
    });
    let logits = p.op(Op::Reshape {
        value: raw,
        shape: v,
    });
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
    let ninf = p.f(f32::NEG_INFINITY);
    let logits = p.op(Op::Select {
        cond: mask,
        a: logits,
        b: ninf,
    });
    let logits = p.op(Op::MaskApply {
        logits,
        mask: words,
    });
    let greedy = p.op(Op::ReduceArgmax(logits));
    let temp = p.f(0.7);
    let scaled = p.op(Op::Div(logits, temp));
    let m = p.op(Op::ReduceMax(scaled));
    let mb = p.op(Op::Broadcast { value: m, shape: v });
    let c = p.op(Op::Sub(scaled, mb));
    let e = p.op(Op::Exp(c));
    let s = p.op(Op::ReduceSum(e));
    let sb = p.op(Op::Broadcast { value: s, shape: v });
    let probs = p.op(Op::Div(e, sb));
    let k = p.op(Op::Const(Literal::U32(40)));
    let keep_k = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::RankLe(k),
    });
    let top_p = p.f(0.9);
    let keep_p = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::CummassLe(top_p),
    });
    let keep = p.op(Op::And(keep_k, keep_p));
    let truncated = p.op(Op::Select {
        cond: keep,
        a: scaled,
        b: ninf,
    });
    let drawn = p.op(Op::RngKeyed {
        state: r,
        shape: v,
        kind: RngKind::Gumbel,
    });
    let perturbed = p.op(Op::Add(truncated, drawn));
    let token = p.op(Op::ReduceArgmax(perturbed));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let greedy1 = p.op(Op::Reshape {
        value: greedy,
        shape: Shape::vector(1),
    });
    let one = p.f(1.0);
    let next = p.op(Op::ScatterAdd {
        base: counts,
        idx: token,
        vals: one,
    });
    let step = p.op(Op::Iota { len: 2 });
    let r_next = p.op(Op::Add(r, step));
    let outs = vec![
        p.output(Shape::vector(1), Dtype::I32, token1),
        p.output(Shape::vector(1), Dtype::I32, greedy1),
        p.output(v, Dtype::F32, next),
        p.output(Shape::vector(2), Dtype::U32, r_next),
        p.output(v, Dtype::Bool, keep_k),
        p.output(v, Dtype::Bool, keep_p),
        p.output(v, Dtype::F32, probs),
        p.output(v, Dtype::F32, drawn),
    ];
    (
        p.container(),
        outs,
        [c_rng, c_counts, c_present, c_mask, c_words],
    )
}

fn sampler_seeds(vocab: u32, seed: u64) -> Vec<(u32, Value)> {
    let n = vocab as usize;
    let mut counts = vec![0.0f32; n];
    let mut present = vec![0.0f32; n];
    let mut mask = vec![true; n];
    let mut words = vec![u32::MAX; n.div_ceil(32)];
    for (i, u) in noise(seed ^ 0x55, 64, 0.0, n as f32)
        .into_iter()
        .enumerate()
    {
        let at = (u as usize).min(n - 1);
        match i % 4 {
            0 => counts[at] += 1.0,
            1 => present[at] = 1.0,
            2 => mask[at] = false,
            _ => words[at / 32] &= !(1 << (at % 32)),
        }
    }
    vec![
        (
            0,
            Value::U32(vec![0x7ce1 ^ seed as u32, 5 + (seed as u32 & 7)]),
        ),
        (1, Value::F32(counts)),
        (2, Value::F32(present)),
        (3, Value::Bool(mask)),
        (4, Value::U32(words)),
    ]
}

/// A sampler over several rows with per-row k and p.
fn rowwise(rows: u32, vocab: u32) -> (TraceContainer, Vec<u32>) {
    let grid = Shape::matrix(rows, vocab);
    let mut p = Program::new();
    let (_, r) = p.input(Shape::vector(2), Dtype::U32);
    let (_, k) = p.input(Shape::vector(rows), Dtype::I32);
    let (_, top_p) = p.input(Shape::vector(rows), Dtype::F32);
    let (_, floor) = p.input(Shape::vector(rows), Dtype::F32);
    let logits = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: grid,
        dtype: Dtype::F32,
    });
    let m = p.op(Op::ReduceMax(logits));
    let mb = p.op(Op::Broadcast {
        value: m,
        shape: grid,
    });
    let c = p.op(Op::Sub(logits, mb));
    let e = p.op(Op::Exp(c));
    let s = p.op(Op::ReduceSum(e));
    let sb = p.op(Op::Broadcast {
        value: s,
        shape: grid,
    });
    let probs = p.op(Op::Div(e, sb));
    let keep_k = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::RankLe(k),
    });
    let keep_p = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::CummassLe(top_p),
    });
    let keep_f = p.op(Op::PivotThreshold {
        input: probs,
        predicate: Predicate::ProbGe(floor),
    });
    let keep = p.op(Op::And(keep_k, keep_p));
    let ninf = p.f(f32::NEG_INFINITY);
    let t = p.op(Op::Select {
        cond: keep,
        a: logits,
        b: ninf,
    });
    let g = p.op(Op::RngKeyed {
        state: r,
        shape: grid,
        kind: RngKind::Gumbel,
    });
    let pert = p.op(Op::Add(t, g));
    let tokens = p.op(Op::ReduceArgmax(pert));
    let outs = vec![
        p.output(Shape::vector(rows), Dtype::I32, tokens),
        p.output(grid, Dtype::Bool, keep_k),
        p.output(grid, Dtype::Bool, keep_p),
        p.output(grid, Dtype::Bool, keep_f),
    ];
    (p.container(), outs)
}

/// engine-cuda's acceptance rule: sort, cumsum, scatter_set, casts, counts.
fn acceptance() -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let (_, entropy) = p.input(Shape::vector(ROWS), Dtype::F32);
    let neg = p.op(Op::Neg(entropy));
    let sorted = p.op(Op::SortDesc(neg));
    let asc = p.op(Op::Neg(sorted));
    let cum = p.op(Op::CumSum(asc));
    let before = p.op(Op::Sub(cum, asc));
    let three = p.f(3.0);
    let within = p.op(Op::Le(before, three));
    let iota = p.op(Op::Iota { len: ROWS });
    let zero = p.op(Op::Const(Literal::U32(0)));
    let none = p.op(Op::Lt(iota, zero));
    let accept = p.op(Op::ScatterSet {
        base: none,
        idx: sorted + 1,
        vals: within,
    });
    let (_, sampled) = p.input(Shape::vector(ROWS), Dtype::I32);
    let (_, noise_ids) = p.input(Shape::vector(ROWS), Dtype::I32);
    let next = p.op(Op::Select {
        cond: accept,
        a: sampled,
        b: noise_ids,
    });
    let as_int = p.op(Op::Cast {
        value: accept,
        dtype: Dtype::I32,
    });
    let count = p.op(Op::ReduceSum(as_int));
    let count1 = p.op(Op::Reshape {
        value: count,
        shape: Shape::vector(1),
    });
    let (_, previous) = p.input(Shape::vector(ROWS), Dtype::I32);
    let same = p.op(Op::Eq(sampled, previous));
    let same_i = p.op(Op::Cast {
        value: same,
        dtype: Dtype::I32,
    });
    let agreed = p.op(Op::ReduceSum(same_i));
    let all = p.op(Op::Const(Literal::I32(ROWS as i32)));
    let unanimous = p.op(Op::Eq(agreed, all));
    let total = p.op(Op::ReduceSum(entropy));
    let n = p.f(ROWS as f32);
    let mean = p.op(Op::Div(total, n));
    let lim = p.f(0.5);
    let calm = p.op(Op::Lt(mean, lim));
    let done = p.op(Op::And(unanimous, calm));
    let done1 = p.op(Op::Reshape {
        value: done,
        shape: Shape::vector(1),
    });
    let outs = vec![
        p.output(Shape::vector(ROWS), Dtype::Bool, accept),
        p.output(Shape::vector(ROWS), Dtype::I32, next),
        p.output(Shape::vector(1), Dtype::I32, count1),
        p.output(Shape::vector(1), Dtype::Bool, done1),
        p.output(Shape::vector(ROWS), Dtype::U32, sorted + 1),
        p.output(Shape::vector(ROWS), Dtype::F32, cum),
    ];
    (p.container(), outs)
}

const N: u32 = 16;

/// The corners: casts at the edges, integer division by zero and by -1,
/// NaN through max/min/argmax/reductions/sorts, gathers and scatters out
/// of range and with duplicates, the structured masks, both draws' bits.
fn corners() -> (TraceContainer, Vec<u32>) {
    let v = Shape::vector(N);
    let mut p = Program::new();
    let (_, x) = p.input(v, Dtype::F32);
    let (_, y) = p.input(v, Dtype::F32);
    let (_, a) = p.input(v, Dtype::I32);
    let (_, b) = p.input(v, Dtype::I32);
    let (_, ua) = p.input(v, Dtype::U32);
    let (_, ub) = p.input(v, Dtype::U32);
    let (_, state) = p.input(Shape::vector(2), Dtype::U32);
    let (_, pos) = p.input(Shape::vector(4), Dtype::U32);
    let (_, table) = p.input(Shape::matrix(5, 3), Dtype::F32);
    let (_, words) = p.input(Shape::vector(1), Dtype::U32);
    let mut outs = Vec::new();
    let mut put = |p: &mut Program, shape: Shape, dtype: Dtype, value: u32| {
        outs.push(p.output(shape, dtype, value));
    };
    for dtype in [Dtype::I32, Dtype::U32, Dtype::Bool] {
        let c = p.op(Op::Cast { value: x, dtype });
        put(&mut p, v, dtype, c);
    }
    let c = p.op(Op::Cast {
        value: a,
        dtype: Dtype::U32,
    });
    put(&mut p, v, Dtype::U32, c);
    let c = p.op(Op::Cast {
        value: ua,
        dtype: Dtype::I32,
    });
    put(&mut p, v, Dtype::I32, c);
    let c = p.op(Op::Cast {
        value: a,
        dtype: Dtype::F32,
    });
    put(&mut p, v, Dtype::F32, c);
    let c = p.op(Op::Cast {
        value: ua,
        dtype: Dtype::F32,
    });
    put(&mut p, v, Dtype::F32, c);
    let bools = p.op(Op::Cast {
        value: a,
        dtype: Dtype::Bool,
    });
    let c = p.op(Op::Cast {
        value: bools,
        dtype: Dtype::I32,
    });
    put(&mut p, v, Dtype::I32, c);
    for op in [
        Op::Div(a, b),
        Op::Rem(a, b),
        Op::Mul(a, b),
        Op::Add(a, b),
        Op::Sub(a, b),
        Op::MaxElem(a, b),
        Op::MinElem(a, b),
    ] {
        let c = p.op(op);
        put(&mut p, v, Dtype::I32, c);
    }
    for op in [
        Op::Div(ua, ub),
        Op::Rem(ua, ub),
        Op::Mul(ua, ub),
        Op::Sub(ua, ub),
        Op::Neg(ua),
        Op::Sign(ua),
    ] {
        let c = p.op(op);
        put(&mut p, v, Dtype::U32, c);
    }
    for op in [Op::Neg(a), Op::Abs(a), Op::Sign(a)] {
        let c = p.op(op);
        put(&mut p, v, Dtype::I32, c);
    }
    for op in [
        Op::MaxElem(x, y),
        Op::MinElem(x, y),
        Op::Sign(x),
        Op::Abs(x),
        Op::Neg(x),
        Op::Rem(x, y),
        Op::Add(x, y),
        Op::Mul(x, y),
        Op::Sub(x, y),
        Op::Div(x, y),
    ] {
        let c = p.op(op);
        put(&mut p, v, Dtype::F32, c);
    }
    for op in [
        Op::Gt(x, y),
        Op::Ge(x, y),
        Op::Eq(x, y),
        Op::Ne(x, y),
        Op::Lt(a, b),
        Op::Le(ua, ub),
    ] {
        let c = p.op(op);
        put(&mut p, v, Dtype::Bool, c);
    }
    let one = Shape::vector(1);
    for op in [Op::ReduceMax(x), Op::ReduceMin(x)] {
        let r = p.op(op);
        let r = p.op(Op::Reshape {
            value: r,
            shape: one,
        });
        put(&mut p, one, Dtype::F32, r);
    }
    for src in [x, a, ua] {
        let r = p.op(Op::ReduceArgmax(src));
        let r = p.op(Op::Reshape {
            value: r,
            shape: one,
        });
        put(&mut p, one, Dtype::I32, r);
    }
    for op in [Op::ReduceSum(a), Op::ReduceMax(a), Op::ReduceMin(a)] {
        let r = p.op(op);
        let r = p.op(Op::Reshape {
            value: r,
            shape: one,
        });
        put(&mut p, one, Dtype::I32, r);
    }
    for op in [Op::ReduceSum(ua), Op::ReduceMax(ua), Op::ReduceMin(ua)] {
        let r = p.op(op);
        let r = p.op(Op::Reshape {
            value: r,
            shape: one,
        });
        put(&mut p, one, Dtype::U32, r);
    }
    let sorted = p.op(Op::SortDesc(x));
    put(&mut p, v, Dtype::F32, sorted);
    put(&mut p, v, Dtype::U32, sorted + 1);
    let top = p.op(Op::TopK { input: y, k: 5 });
    put(&mut p, Shape::vector(5), Dtype::F32, top);
    put(&mut p, Shape::vector(5), Dtype::U32, top + 1);
    // Gathers: rows of a table by indices in and out of range.
    let g = p.op(Op::Gather { src: table, idx: a });
    put(&mut p, Shape::matrix(N, 3), Dtype::F32, g);
    let g = p.op(Op::Gather { src: x, idx: ua });
    put(&mut p, v, Dtype::F32, g);
    let cols = p.op(Op::Cast {
        value: a,
        dtype: Dtype::U32,
    });
    let first5 = p.op(Op::Iota { len: 5 });
    let cols5 = p.op(Op::Gather {
        src: cols,
        idx: first5,
    });
    let g = p.op(Op::GatherRow {
        src: table,
        idx: cols5,
    });
    put(&mut p, Shape::vector(5), Dtype::F32, g);
    // Scatters with duplicate and out-of-range indices.
    let s = p.op(Op::ScatterSet {
        base: x,
        idx: a,
        vals: y,
    });
    put(&mut p, v, Dtype::F32, s);
    let s = p.op(Op::ScatterAdd {
        base: x,
        idx: b,
        vals: y,
    });
    put(&mut p, v, Dtype::F32, s);
    let s = p.op(Op::ScatterAdd {
        base: ua,
        idx: a,
        vals: ub,
    });
    put(&mut p, v, Dtype::U32, s);
    let seven = p.op(Op::Const(Literal::I32(7)));
    let s = p.op(Op::ScatterSet {
        base: a,
        idx: b,
        vals: seven,
    });
    put(&mut p, v, Dtype::I32, s);
    // Structured masks.
    let mask_shape = Shape::matrix(4, 12);
    for op in [
        Op::CausalMask {
            positions: pos,
            len: 12,
        },
        Op::SlidingWindowMask {
            positions: pos,
            len: 12,
            window: 3,
        },
        Op::SinkWindowMask {
            positions: pos,
            len: 12,
            sink: 2,
            window: 3,
        },
    ] {
        let m = p.op(op);
        put(&mut p, mask_shape, Dtype::Bool, m);
    }
    // Packed mask, transpose, matmul, cumsum/prod, the maps.
    let m = p.op(Op::MaskApply {
        logits: x,
        mask: words,
    });
    put(&mut p, v, Dtype::F32, m);
    let t = p.op(Op::Transpose(table));
    put(&mut p, Shape::matrix(3, 5), Dtype::F32, t);
    let mm = p.op(Op::MatMul(table, t));
    put(&mut p, Shape::matrix(5, 5), Dtype::F32, mm);
    let cs = p.op(Op::CumSum(table));
    put(&mut p, Shape::matrix(5, 3), Dtype::F32, cs);
    let cp = p.op(Op::CumProd(table));
    put(&mut p, Shape::matrix(5, 3), Dtype::F32, cp);
    // Integer scans wrap, as the interpreter's do.
    for (value, dtype) in [(a, Dtype::I32), (ua, Dtype::U32)] {
        for op in [Op::CumSum(value), Op::CumProd(value)] {
            let c = p.op(op);
            put(&mut p, v, dtype, c);
        }
    }
    let pos_y = p.op(Op::Abs(y));
    for op in [
        Op::Exp(y),
        Op::Log(pos_y),
        Op::Sqrt(pos_y),
        Op::Rsqrt(pos_y),
        Op::Recip(y),
        Op::Sin(y),
        Op::Cos(y),
    ] {
        let c = p.op(op);
        put(&mut p, v, Dtype::F32, c);
    }
    let bcast = p.op(Op::Broadcast {
        value: pos,
        shape: Shape::matrix(4, 6),
    });
    put(&mut p, Shape::matrix(4, 6), Dtype::U32, bcast);
    // Draws: the uniform's bits are exact, Gumbel and normal within rounding.
    for kind in [RngKind::Uniform, RngKind::Gumbel, RngKind::Normal] {
        let d = p.op(Op::RngKeyed {
            state,
            shape: Shape::matrix(3, 50),
            kind,
        });
        put(&mut p, Shape::matrix(3, 50), Dtype::F32, d);
        let d = p.op(Op::Rng {
            stream: 11,
            shape: Shape::vector(40),
            kind,
        });
        put(&mut p, Shape::vector(40), Dtype::F32, d);
    }
    (p.container(), outs)
}

fn corner_seeds() -> Vec<(u32, Value)> {
    let x = vec![
        1.5,
        -2.5,
        f32::NAN,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        3e9,
        -3e9,
        5e9,
        2.2e9,
        7.0,
        7.0,
        -1.0,
        0.49,
        1e-30,
    ];
    let y = vec![
        2.0,
        3.0,
        1.0,
        -0.0,
        0.0,
        1.0,
        f32::NAN,
        7.0,
        7.0,
        -2.0,
        0.5,
        7.0,
        8.0,
        1.0,
        7.0,
        2.0,
    ];
    let a = vec![
        7,
        -7,
        i32::MIN,
        5,
        0,
        3,
        -1,
        2,
        9,
        16,
        -3,
        -1,
        i32::MIN,
        4,
        i32::MAX,
        15,
    ];
    // Division corners, beside the others: MIN / -1 wraps, x / 0 is 0,
    // MIN / MIN, -1 / MAX.
    let b = vec![2, 2, -1, 0, 3, 3, 0, -2, 4, 1, 1, i32::MAX, i32::MIN, -5, -1, 15];
    let ua = vec![
        7,
        u32::MAX,
        0x8000_0000,
        5,
        0,
        3,
        1,
        2,
        9,
        16,
        3,
        u32::MAX,
        u32::MAX,
        40,
        12,
        15,
    ];
    // u32::MAX / 1, u32::MAX / u32::MAX.
    let ub = vec![2, 2, 0, 0, 3, 7, 0xFFFF_FFF0, 2, 4, 1, 1, 1, u32::MAX, 5, 3, 15];
    vec![
        (0, Value::F32(x)),
        (1, Value::F32(y)),
        (2, Value::I32(a)),
        (3, Value::I32(b)),
        (4, Value::U32(ua)),
        (5, Value::U32(ub)),
        (6, Value::U32(vec![0xdead_beef, 3])),
        (7, Value::U32(vec![0, 4, 9, 11])),
        (
            8,
            Value::F32((0..15).map(|i| 0.25 + i as f32 * 0.5).collect()),
        ),
        (
            9,
            Value::U32(vec![0b1010_0110_1111_0000_1100_1010_0101_1011]),
        ),
    ]
}

// ------------------------------------------------------------------ tests

#[test]
fn an_epilogue_agrees_with_the_host_interpreter_every_case() {
    let Some(dev) = device() else {
        return;
    };
    let device = &dev.device;
    let mut stages = Stages::default();

    the_statistics_agree(device, &mut stages);
    a_flat_top_k_agrees(device, &mut stages);
    a_nucleus_cut_agrees(device, &mut stages);
    a_rank_cut_agrees(device, &mut stages);
    the_gumbel_draws_agree(device, &mut stages);
    the_bf16_rules_agree(device, &mut stages);
    a_near_tie_draws_evenly(device, &mut stages);
    a_cast_rounds_a_plane_by_what_it_may_hold(device, &mut stages);
    the_acceptance_rule_agrees(device, &mut stages);
    the_corners_agree(device, &mut stages);
    the_sampler_agrees(device, &mut stages, VOCAB);
    the_rowwise_sampler_agrees(device, &mut stages);
    a_group_answers_what_each_lane_answers_alone(device, &mut stages);
    a_guest_reads_no_row_past_its_own_lanes(device, &mut stages);
    carried_cells_answer_what_the_rings_answer(device, &mut stages);
    eprintln!("compiled {} stage program(s)", stages.len());
    // Last: it seals the device.
    registration_compiles_every_pass(device);
}

/// Registration's compile (`Stages::compile_all`, every lane rung and every
/// plane a fire returns) leaves no pass anything to compile: four passes of
/// a carrying group after the device seals build no program; and the
/// program's fire facts cover every rung it compiled.
fn registration_compiles_every_pass(device: &Device) {
    let (container, out) = carrying(VOCAB);
    let plan = plan_of(container, VOCAB);
    let carried = engine_xla::guest::carried_channels(&plan.package);
    let key = key_of(&plan);
    let lanes = 3u32;
    let probe = kept(device, &logits_plane(lanes, VOCAB, 1), lanes, vec![(0, 1)]);
    let read = &probe.planes[0];
    let words = engine_xla::guest::carried_words(&plan.package);
    let per_lane: u32 = words
        .iter()
        .map(|&(_, w)| (engine_xla::guest::lower::cell_rows(w as usize) as u32).next_multiple_of(2))
        .sum();
    let pool_rows = per_lane * lanes + 8;
    let mut stages = Stages::default();
    stages.top = lanes;
    stages.pool_rows = i64::from(pool_rows);
    stages.sink_row = pool_rows - 8;
    stages.planes = vec![(
        read.name.clone(),
        read.rows,
        read.width,
        read.layers,
        read.readout,
        read.dtype,
        read.normal,
    )];
    // A program registers on a served model, its load's ladder sealed:
    // registration compiles are no ladder misses.
    device.seal();
    let before = device.misses();
    stages
        .compile_all(device, &plan.package, key, &carried)
        .expect("every stage compiles at registration");
    assert_eq!(device.misses(), before, "registration compiles are not ladder misses");
    let compiled = stages.len();
    let facts = stages.fire_facts(key, &carried, device.load_temp());
    eprintln!("registration compiled {compiled} stage program(s); fire facts {facts:?}");
    assert_eq!(facts.max_lanes, 4, "every lane rung up to the widest fire");
    assert!(
        facts.fixed + 4 * facts.a_lane > 0,
        "a pass returns its outputs: {facts:?}"
    );
    let seeds: Vec<Vec<(u32, Value)>> = (0..lanes)
        .map(|i| {
            vec![
                (0, Value::U32(vec![9 + i, 1])),
                (1, Value::F32(vec![0.0; VOCAB as usize])),
            ]
        })
        .collect();
    let mut insts: Vec<InterpInstance> = seeds.iter().map(|s| instance(&plan, s)).collect();
    let cells: Vec<Vec<(u32, u32)>> = (0..lanes)
        .map(|l| {
            let mut at = l * per_lane;
            words
                .iter()
                .map(|&(c, w)| {
                    let first = at;
                    at += (engine_xla::guest::lower::cell_rows(w as usize) as u32).next_multiple_of(2);
                    (c, first)
                })
                .collect()
        })
        .collect();
    let mut pool = device
        .zeros_or_dry(dtype::Dtype::U32, pool_rows, 128)
        .expect("the pool");
    let _ = engine_xla::guest::run::cells_program(device, i64::from(pool_rows));
    let before = before.max(device.misses());
    {
        let writes: Vec<(&InterpInstance, &[(u32, u32)])> =
            insts.iter().zip(&cells).map(|(i, c)| (i, c.as_slice())).collect();
        engine_xla::guest::run::write_cells(
            device,
            &plan,
            &mut pool,
            i64::from(pool_rows),
            stages.sink_row,
            &writes,
        )
        .expect("the seeds land");
    }
    for pass in 0..4 {
        let plane = logits_plane(lanes, VOCAB, 90 + pass);
        let kept = kept(device, &plane, lanes, (0..lanes).map(|l| (l, 1)).collect());
        let mut members: Vec<Member<'_>> = insts
            .iter_mut()
            .enumerate()
            .map(|(l, inst)| Member {
                inst,
                bound: bound(&plan, &kept, l),
                cells: cells[l].clone(),
            })
            .collect();
        let outcomes = step_group(
            device,
            &mut stages,
            key,
            &plan,
            Some(&kept),
            &mut members,
            &carried,
            &mut pool,
            eta_ir::registry::Stage::Epilogue,
        );
        assert!(outcomes.iter().all(|o| *o == StepOutcome::Committed), "{outcomes:?}");
        for inst in &insts {
            let _ = take_all(&plan, inst, &[out]);
        }
    }
    assert_eq!(stages.len(), compiled, "no pass compiled a stage");
    assert_eq!(device.misses(), before, "no program built after the seal");
}

fn the_statistics_agree(device: &Device, stages: &mut Stages) {
    let (container, outs) = statistics();
    let plan = plan_of(container, VOCAB);
    let seeds = vec![
        (0, Value::F32(vec![0.7])),
        (
            1,
            Value::I32(
                (0..ROWS as i32)
                    .map(|r| (r * 131 + 17) % VOCAB as i32)
                    .collect(),
            ),
        ),
    ];
    let plane = logits_plane(ROWS, VOCAB, 7);
    let want = host(&plan, &seeds, &plane, ROWS, &outs);
    // The lane's rows sit after two other lanes' in the readout.
    let mut shared = logits_plane(2, VOCAB, 99);
    shared.extend_from_slice(&plane);
    let kept = kept(device, &shared, ROWS + 2, vec![(0, 2), (2, ROWS)]);
    let got = on_device(
        device,
        stages,
        &plan,
        &seeds,
        Some(&kept),
        bound(&plan, &kept, 1),
        &outs,
    );
    close(&got[0], &want[0], 1e-5, "gather_row of the scaled logits");
    exact(&got[1], &want[1], "argmax per row");
    exact(&got[2], &want[2], "top-4 ids per row");
    close(&got[3], &want[3], 1e-5, "top-4 values");
    close(&got[4], &want[4], 1e-4, "softmax row mass");
    close(&got[5], &want[5], 1e-5, "probability at the argmax");
}

/// A flat top-k breaks its ties as the interpreter's order does: equal
/// values in index order, -0 equal to +0 (each keeping its own bits), NaN
/// after every number.
fn a_flat_top_k_agrees(device: &Device, stages: &mut Stages) {
    let (container, outs) = flat_top_k();
    let plan = plan_of(container, VOCAB);
    let mut plane = logits_plane(ROWS, VOCAB, 11);
    let n = plane.len();
    for x in &mut plane {
        *x -= 20.0;
    }
    // ties at the top across rows, a signed-zero pair above them, NaNs
    for at in [n - 3, 5, 3 * VOCAB as usize + 1, 9, 6 * VOCAB as usize] {
        plane[at] = -1.5;
    }
    plane[2 * VOCAB as usize + 7] = 0.0;
    plane[VOCAB as usize + 3] = -0.0;
    for at in [0, 4 * VOCAB as usize + 2, n - 1] {
        plane[at] = f32::NAN;
    }
    let want = host(&plan, &[], &plane, ROWS, &outs);
    let kept = kept(device, &plane, ROWS, vec![(0, ROWS)]);
    let got = on_device(
        device,
        stages,
        &plan,
        &[],
        Some(&kept),
        bound(&plan, &kept, 0),
        &outs,
    );
    exact(&got[0], &want[0], "flat top-4 ids");
    exact(&got[1], &want[1], "flat top-4 values");
    exact(&got[2], &want[2], "flat top-9 ids");
    exact(&got[3], &want[3], "flat top-9 values");
    exact(&got[4], &want[4], "flat top-1 id");
    exact(&got[5], &want[5], "flat top-1 value");
}

/// `gumbel_argmax` answers the unfused draw over f32 logits bit for bit, on
/// the device and in the interpreter, at every seed and scale (the host's
/// rounded 1 / T, 1.0 / 0.7 among them); over bf16 logits it answers the
/// interpreter.
fn the_gumbel_draws_agree(device: &Device, stages: &mut Stages) {
    let (container, outs) = gumbel_draws();
    let plan = plan_of(container, VOCAB);
    let plane = logits_plane(ROWS, VOCAB, 23);
    let kept = kept(device, &plane, ROWS, vec![(0, ROWS)]);
    for seed in 0..12u32 {
        let seeds = vec![
            (0, Value::F32(vec![1.0 / (0.6 + 0.1 * (seed % 5) as f32)])),
            (1, Value::U32(vec![0x9E37_79B9 ^ (seed * 7919), seed])),
        ];
        let want = host(&plan, &seeds, &plane, ROWS, &outs);
        let got = on_device(
            device,
            stages,
            &plan,
            &seeds,
            Some(&kept),
            bound(&plan, &kept, 0),
            &outs,
        );
        exact(
            &want[0],
            &want[1],
            "the interpreter's fused and unfused draws",
        );
        exact(&got[0], &want[0], "the fused draw");
        exact(&got[1], &want[1], "the unfused draw");
        exact(&got[2], &want[2], "the fused draw over bf16 logits");
    }
}

fn the_bf16_rules_agree(device: &Device, stages: &mut Stages) {
    let (container, outs) = bf16_rules();
    let plan = plan_of(container, VOCAB);
    let plane = logits_plane(ROWS, VOCAB, 29);
    let want = host(&plan, &[], &plane, ROWS, &outs);
    let kept = kept(device, &plane, ROWS, vec![(0, ROWS)]);
    let got = on_device(
        device,
        stages,
        &plan,
        &[],
        Some(&kept),
        bound(&plan, &kept, 0),
        &outs,
    );
    exact(&got[0], &want[0], "a bf16 product, rounded once");
    close(&got[1], &want[1], 1e-2, "a bf16 row's f32 sum");
    // each prefix the f32 running sum rounded once: an f32 order apart at
    // most one bf16 step
    let half = |v: &Value| -> Vec<f32> {
        match v {
            Value::Bf16(h) => h
                .iter()
                .map(|&b| f32::from_bits(u32::from(b) << 16))
                .collect(),
            other => panic!("a bf16 scan answered {other:?}"),
        }
    };
    let (g, w) = (half(&got[2]), half(&want[2]));
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert!(
            (a - b).abs() <= b.abs() / 128.0 + 1e-3,
            "scan lane {i}: {a} against {b}"
        );
    }
    exact(&got[3], &want[3], "a bf16 top-4's ids");
    exact(&got[4], &want[4], "a bf16 top-4's values");
}

/// Four exactly equal largest logits a row (bf16, where a bf16 store of the
/// perturbed lane would collapse them onto the lowest id) are drawn evenly.
fn a_near_tie_draws_evenly(device: &Device, stages: &mut Stages) {
    let (container, outs) = gumbel_draws();
    let plan = plan_of(container, VOCAB);
    let mut plane = vec![-8.0f32; (ROWS * VOCAB) as usize];
    let tied = [11usize, 1000, 2047, 3999];
    for row in 0..ROWS as usize {
        for &t in &tied {
            plane[row * VOCAB as usize + t] = 16.0;
        }
    }
    let kept = kept(device, &plane, ROWS, vec![(0, ROWS)]);
    let mut counts = [0u32; 4];
    for seed in 0..64u32 {
        let seeds = vec![
            (0, Value::F32(vec![1.0])),
            (1, Value::U32(vec![seed.wrapping_mul(2_654_435_761), 1])),
        ];
        let got = on_device(
            device,
            stages,
            &plan,
            &seeds,
            Some(&kept),
            bound(&plan, &kept, 0),
            &outs,
        );
        let Value::I32(ids) = &got[2] else {
            panic!("ids")
        };
        for &id in ids {
            let at = tied
                .iter()
                .position(|&t| t == id as usize)
                .expect("a tied id");
            counts[at] += 1;
        }
    }
    // 512 draws over 4 ties: 128 each, sd ~9.8
    assert!(
        counts.iter().all(|&c| (96..=160).contains(&c)),
        "near-tie draws {counts:?}"
    );
}

/// A bf16 cast of a plane rounds as `bf16_bits` does whichever form the
/// lowering takes: a plane the engine does not state free of denormals
/// (host-written here, denormals of both signs and at the bf16 halfway
/// among them) takes the bit form and keeps them; a plane it states free
/// of them (a matmul's) takes `reduce_precision`.
fn a_cast_rounds_a_plane_by_what_it_may_hold(device: &Device, stages: &mut Stages) {
    let (container, outs) = bf16_cast();
    let plan = plan_of(container, VOCAB);
    let mut plane = logits_plane(ROWS, VOCAB, 41);
    let free = plane.clone();
    for (i, x) in [
        1e-40f32,
        -3.0e-39,
        5.877_472e-39,
        f32::from_bits(0x0000_8000),
        -1.0e-45,
    ]
    .into_iter()
    .enumerate()
    {
        plane[i * 997 + 3] = x;
    }
    let want = host(&plan, &[], &plane, ROWS, &outs);
    let may = kept(device, &plane, ROWS, vec![(0, ROWS)]);
    let got = on_device(
        device,
        stages,
        &plan,
        &[],
        Some(&may),
        bound(&plan, &may, 0),
        &outs,
    );
    exact(
        &got[0],
        &want[0],
        "a bf16 cast of a plane that may hold denormals",
    );
    let want = host(&plan, &[], &free, ROWS, &outs);
    let mut stated = kept(device, &free, ROWS, vec![(0, ROWS)]);
    stated.planes = stated
        .planes
        .into_iter()
        .map(|p| p.with_normal(true))
        .collect();
    let got = on_device(
        device,
        stages,
        &plan,
        &[],
        Some(&stated),
        bound(&plan, &stated, 0),
        &outs,
    );
    exact(
        &got[0],
        &want[0],
        "a bf16 cast of a plane stated free of denormals",
    );
}

/// The nucleus cut (`cummass_le(p)`) over crafted rows, a p a row: a plain
/// softmax row, a tie straddling the cut, zeros of both signs past it, NaN
/// lanes, a cut landing exactly on a running sum, p of 0 and p past the
/// row's mass; with no negative lane anywhere (the cut found by bisection)
/// and with one negative row (the order and its verdicts sorted back).
fn nucleus_cut() -> (TraceContainer, Vec<u32>) {
    let grid = Shape::matrix(ROWS, VOCAB);
    let mut p = Program::new();
    let (_, top_p) = p.input(Shape::vector(ROWS), Dtype::F32);
    let x = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: grid,
        dtype: Dtype::F32,
    });
    let keep = p.op(Op::PivotThreshold {
        input: x,
        predicate: Predicate::CummassLe(top_p),
    });
    let outs = vec![p.output(grid, Dtype::Bool, keep)];
    (p.container(), outs)
}

fn nucleus_plane(negative_row: bool) -> Vec<f32> {
    let n = VOCAB as usize;
    let mut plane = vec![0.0f32; ROWS as usize * n];
    let mut row = |r: usize, f: &dyn Fn(usize) -> f32| {
        for i in 0..n {
            plane[r * n + i] = f(i);
        }
    };
    // 0: a softmax row
    let logits = logits_plane(1, VOCAB, 3);
    let mx = logits.iter().copied().fold(f32::MIN, f32::max);
    let e: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let sum: f32 = e.iter().sum();
    row(0, &|i| e[i] / sum);
    // 1: 100 equal lanes scattered over the row, the cut inside them
    row(1, &|i| {
        if i % 37 == 5 && i / 37 < 100 {
            0.01
        } else {
            0.0
        }
    });
    // 2: ten lanes of 0.1, then zeros of both signs (one key, by index)
    row(2, &|i| {
        if i % 400 == 17 && i / 400 < 10 {
            0.1
        } else if i % 2 == 0 {
            0.0
        } else {
            -0.0
        }
    });
    // 3: a few lanes and NaNs past them
    row(3, &|i| match i {
        3 => 0.4,
        900 => 0.3,
        1200 | 7 | 4000 => f32::NAN,
        _ => 0.0,
    });
    // 4: signed lanes (a negative row), else a second softmax row
    if negative_row {
        let l = logits_plane(1, VOCAB, 4);
        row(4, &|i| l[i] * 0.01);
    } else {
        row(4, &|i| e[(i * 7) % n] / sum);
    }
    // 5, 6: the softmax row at p 0 and p past its mass
    row(5, &|i| e[i] / sum);
    row(6, &|i| e[i] / sum);
    // 7: 0.5, 0.25, 0.25: the running sum lands on p exactly
    row(7, &|i| match i {
        10 => 0.5,
        20 | 30 => 0.25,
        _ => 0.0,
    });
    plane
}

fn rank_cut() -> (TraceContainer, Vec<u32>) {
    let grid = Shape::matrix(ROWS, VOCAB);
    let mut p = Program::new();
    let (_, k) = p.input(Shape::vector(ROWS), Dtype::U32);
    let x = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: grid,
        dtype: Dtype::F32,
    });
    let keep = p.op(Op::PivotThreshold {
        input: x,
        predicate: Predicate::RankLe(k),
    });
    let outs = vec![p.output(grid, Dtype::Bool, keep)];
    (p.container(), outs)
}

/// Rows for the top-k cut's corners, each with its `k` (`rank_ks`).
fn rank_plane() -> Vec<f32> {
    let n = VOCAB as usize;
    let mut plane = vec![0.0f32; ROWS as usize * n];
    let mut row = |r: usize, f: &dyn Fn(usize) -> f32| {
        for i in 0..n {
            plane[r * n + i] = f(i);
        }
    };
    let logits = logits_plane(1, VOCAB, 5);
    // 0: a logits row (both signs), k 40
    row(0, &|i| logits[i]);
    // 1: three lanes above five tied at the k-th key, k 5: every tie kept
    row(1, &|i| match i {
        9 | 900 | 2000 => 3.0,
        _ if i % 500 == 11 && i / 500 < 5 => 1.5,
        _ => -1.0,
    });
    // 2: NaNs above and among a few lanes, the rest -inf, k 6
    row(2, &|i| match i {
        0 | 50 | 3000 => f32::NAN,
        7 => 2.0,
        70 => 1.0,
        700 => -0.5,
        _ => f32::NEG_INFINITY,
    });
    // 3: zeros of both signs (one key) under two lanes, k 3
    row(3, &|i| match i {
        100 => 0.25,
        200 => -0.0,
        300 => 1.0,
        _ if i % 2 == 0 => 0.0,
        _ => -0.0,
    });
    // 4: the logits row, k 0: none kept
    row(4, &|i| logits[i]);
    // 5: the logits row, k past the vocabulary: every lane kept
    row(5, &|i| logits[i]);
    // 6: NaNs, k past the real lanes: every real lane kept
    row(6, &|i| if i % 3 == 0 { logits[i] } else { f32::NAN });
    // 7: a negative row, k 1: its largest only
    row(7, &|i| -1.0 - (i as f32) * 1e-3);
    plane
}

fn a_rank_cut_agrees(device: &Device, stages: &mut Stages) {
    let (container, outs) = rank_cut();
    let plan = plan_of(container, VOCAB);
    let ks = vec![40, 5, 6, 3, 0, VOCAB + 7, VOCAB, 1];
    let seeds = vec![(0, Value::U32(ks))];
    let plane = rank_plane();
    let want = host(&plan, &seeds, &plane, ROWS, &outs);
    let kept = kept(device, &plane, ROWS, vec![(0, ROWS)]);
    let got = on_device(
        device,
        stages,
        &plan,
        &seeds,
        Some(&kept),
        bound(&plan, &kept, 0),
        &outs,
    );
    exact(&got[0], &want[0], "the top-k cut (bisection)");
}

fn a_nucleus_cut_agrees(device: &Device, stages: &mut Stages) {
    let (container, outs) = nucleus_cut();
    let plan = plan_of(container, VOCAB);
    let ps = vec![0.9, 0.505, 1.5, 2.0, 0.8, 0.0, 1.5, 0.75];
    let seeds = vec![(0, Value::F32(ps))];
    for negative_row in [false, true] {
        let plane = nucleus_plane(negative_row);
        let want = host(&plan, &seeds, &plane, ROWS, &outs);
        let kept = kept(device, &plane, ROWS, vec![(0, ROWS)]);
        let got = on_device(
            device,
            stages,
            &plan,
            &seeds,
            Some(&kept),
            bound(&plan, &kept, 0),
            &outs,
        );
        exact(
            &got[0],
            &want[0],
            if negative_row {
                "the nucleus cut, a negative row (sorted back)"
            } else {
                "the nucleus cut, no negative lane (bisection)"
            },
        );
    }
}

fn the_acceptance_rule_agrees(device: &Device, stages: &mut Stages) {
    let (container, outs) = acceptance();
    let plan = plan_of(container, VOCAB);
    let seeds = vec![
        (
            0,
            Value::F32(vec![0.9, 0.05, 2.5, 0.4, 1.7, 0.01, 0.6, 3.2]),
        ),
        (1, Value::I32((0..ROWS as i32).map(|r| 100 + r).collect())),
        (
            2,
            Value::I32((0..ROWS as i32).map(|r| 900 + 7 * r).collect()),
        ),
        (
            3,
            Value::I32(
                (0..ROWS as i32)
                    .map(|r| if r == 3 { -1 } else { 100 + r })
                    .collect(),
            ),
        ),
    ];
    let want = host(&plan, &seeds, &[], 0, &outs);
    let got = on_device(device, stages, &plan, &seeds, None, Vec::new(), &outs);
    for i in 0..5 {
        exact(&got[i], &want[i], &format!("acceptance output {i}"));
    }
    close(&got[5], &want[5], 1e-6, "the cumulative sum");
}

fn the_corners_agree(device: &Device, stages: &mut Stages) {
    let (container, outs) = corners();
    let plan = plan_of(container, VOCAB);
    let seeds = corner_seeds();
    let want = host(&plan, &seeds, &[], 0, &outs);
    let got = on_device(device, stages, &plan, &seeds, None, Vec::new(), &outs);
    let mut floats = 0;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        match w {
            Value::F32(_) => {
                // Transcendentals and float division round on the device.
                close(g, w, 2e-6, &format!("corner output {i}"));
                floats += 1;
            }
            _ => exact(g, w, &format!("corner output {i}")),
        }
    }
    // The uniform draws are bit-exact.
    let n = outs.len();
    exact(&got[n - 6], &want[n - 6], "the keyed uniform draw");
    exact(&got[n - 5], &want[n - 5], "the stream uniform draw");
    assert!(floats > 0);
}

fn the_sampler_agrees(device: &Device, stages: &mut Stages, vocab: u32) {
    let (container, outs, _) = sampler(vocab);
    let plan = plan_of(container, vocab);
    for seed in [3u64, 4, 5] {
        let seeds = sampler_seeds(vocab, seed);
        let plane = logits_plane(1, vocab, seed);
        let want = host(&plan, &seeds, &plane, 1, &outs);
        let kept = kept(device, &plane, 1, vec![(0, 1)]);
        let got = on_device(
            device,
            stages,
            &plan,
            &seeds,
            Some(&kept),
            bound(&plan, &kept, 0),
            &outs,
        );
        exact(&got[0], &want[0], "the sampled token");
        exact(&got[1], &want[1], "the greedy token");
        exact(&got[2], &want[2], "the next histogram");
        exact(&got[3], &want[3], "the next rng state");
        exact(&got[4], &want[4], "the top-k keep");
        exact(&got[5], &want[5], "the top-p keep");
        close(&got[6], &want[6], 1e-5, "the probabilities");
        close(&got[7], &want[7], 1e-5, "the Gumbel draw");
    }
}

fn the_rowwise_sampler_agrees(device: &Device, stages: &mut Stages) {
    let rows = 4;
    let (container, outs) = rowwise(rows, VOCAB);
    let plan = plan_of(container, VOCAB);
    let seeds = vec![
        (0, Value::U32(vec![17, 2])),
        (1, Value::I32(vec![1, 50, 0, 5000])),
        (2, Value::F32(vec![0.5, 0.95, 1.0, 0.0])),
        (3, Value::F32(vec![1e-4, 1e-3, 0.0, 0.5])),
    ];
    let plane = logits_plane(rows, VOCAB, 11);
    let want = host(&plan, &seeds, &plane, rows, &outs);
    let kept = kept(device, &plane, rows, vec![(0, rows)]);
    let got = on_device(
        device,
        stages,
        &plan,
        &seeds,
        Some(&kept),
        bound(&plan, &kept, 0),
        &outs,
    );
    exact(&got[0], &want[0], "the sampled tokens");
    exact(&got[1], &want[1], "per-row top-k");
    exact(&got[2], &want[2], "per-row top-p");
    exact(&got[3], &want[3], "per-row floor");
}

fn a_group_answers_what_each_lane_answers_alone(device: &Device, stages: &mut Stages) {
    let (container, outs, _) = sampler(VOCAB);
    let plan = plan_of(container, VOCAB);
    let lanes = 5u32;
    let plane = logits_plane(lanes, VOCAB, 21);
    let kept = kept(device, &plane, lanes, (0..lanes).map(|l| (l, 1)).collect());
    // Three instances, reading lanes 4, 1 and 2 of the readout.
    let seated = [4usize, 1, 2];
    let seeds: Vec<Vec<(u32, Value)>> = (0..3).map(|i| sampler_seeds(VOCAB, 40 + i)).collect();
    let mut insts: Vec<InterpInstance> = seeds.iter().map(|s| instance(&plan, s)).collect();
    let key = key_of(&plan);
    let mut members: Vec<Member<'_>> = insts
        .iter_mut()
        .zip(seated)
        .map(|(inst, lane)| Member {
            inst,
            bound: bound(&plan, &kept, lane),
            cells: Vec::new(),
        })
        .collect();
    let outcomes = step_group(
        device,
        stages,
        key,
        &plan,
        Some(&kept),
        &mut members,
        &[],
        &mut None,
        eta_ir::registry::Stage::Epilogue,
    );
    assert!(
        outcomes.iter().all(|o| *o == StepOutcome::Committed),
        "{outcomes:?}"
    );
    for (i, lane) in seated.iter().enumerate() {
        let row = &plane[*lane * VOCAB as usize..(*lane + 1) * VOCAB as usize];
        let want = host(&plan, &seeds[i], row, 1, &outs);
        let got = take_all(&plan, &insts[i], &outs);
        for j in 0..6 {
            exact(&got[j], &want[j], &format!("group lane {i} output {j}"));
        }
        close(&got[6], &want[6], 1e-5, "group probabilities");
    }
}

/// Each row's largest logit over `rows` rows of the lane `lane` the
/// program is attached at (`lane` past the first, a lane after it).
fn row_peaks(rows: u32, lane: u32) -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let logits = p.op(Op::Output {
        index: 0,
        lane,
        shape: Shape::matrix(rows, VOCAB),
        dtype: Dtype::F32,
    });
    let peak = p.op(Op::ReduceMax(logits));
    let peak = p.op(Op::Reshape {
        value: peak,
        shape: Shape::vector(rows),
    });
    let out = p.output(Shape::vector(rows), Dtype::F32, peak);
    (p.container(), vec![out])
}

/// A guest reads its own lanes' rows and no other request's: lane 1's row
/// (a sentinel) lies right after lane 0's, and a program attached at lane 0
/// alone that reads two rows is refused by name, its channel holding
/// nothing; attached over both lanes it reads both rows. A program reading
/// a lane past the ones it is attached over is refused before it binds.
fn a_guest_reads_no_row_past_its_own_lanes(device: &Device, stages: &mut Stages) {
    const SENTINEL: f32 = 4321.0;
    let mut plane = logits_plane(2, VOCAB, 61);
    plane[VOCAB as usize..].fill(SENTINEL);
    let kept = kept(device, &plane, 2, vec![(0, 1), (1, 1)]);
    let (container, outs) = row_peaks(2, 0);
    let plan = plan_of(container, VOCAB);

    let alone = bound(&plan, &kept, 0);
    assert_eq!(
        alone[0].seat,
        Seat { first: 0, count: 1 },
        "a lane attached alone is seated on its own row, not its neighbour's"
    );
    let mut inst = instance(&plan, &[]);
    let mut runner = OnDevice::new(device, stages, key_of(&plan), Some(&kept), alone);
    let outcome = eta_exec::step_with(
        &mut inst,
        &plan,
        &PassInputs::none(),
        &mut runner,
        eta_ir::registry::Stage::Epilogue,
    );
    match &outcome {
        StepOutcome::Faulted(why) => assert!(
            why.contains("reads 2 rows of output `logits` and its lane holds 1"),
            "refused for reading past its lane: {why}"
        ),
        other => panic!("a pass reading its neighbour's row was not refused: {other:?}"),
    }
    for &c in &outs {
        let held = eta_exec::host_take(&inst, &plan, c);
        assert!(
            !matches!(&held.1, Some(Value::F32(x)) if x.contains(&SENTINEL)),
            "the refused pass put its neighbour's row on channel {c}: {held:?}"
        );
    }

    // Attached over both lanes, its two rows are its own.
    let both = engine_xla::guest::run::bound(&plan, Some(&kept), 0, 2)
        .expect("the program reads its own lanes")
        .expect("the lanes bind");
    assert_eq!(both[0].seat, Seat { first: 0, count: 2 });
    let got = on_device(device, stages, &plan, &[], Some(&kept), both, &outs);
    let want = host(&plan, &[], &plane, 2, &outs);
    exact(&got[0], &want[0], "the peaks of the attachment's two rows");
    assert!(
        matches!(&got[0], Value::F32(x) if x[1] == SENTINEL),
        "the second lane's row is the attachment's own: {got:?}"
    );

    // A program reading its lane 1, attached over lane 0 alone.
    let (container, _) = row_peaks(1, 1);
    let plan = plan_over(container, VOCAB, 2);
    let refused = engine_xla::guest::run::bound(&plan, Some(&kept), 0, 1)
        .expect_err("a program reading a lane it is not attached over binds nothing");
    assert!(
        refused.contains("reads output `logits` of its lane 1, and it is attached over 1 lane(s)"),
        "refused by name: {refused}"
    );
}

/// A sampler that carries its rng state and histogram from pass to pass in
/// rings only it touches.
fn carrying(vocab: u32) -> (TraceContainer, u32) {
    let v = Shape::vector(vocab);
    let mut p = Program::new();
    let rng_chan = p.channels.len() as u32;
    p.channels.push(ChannelDecl {
        shape: Shape::vector(2),
        dtype: ChanDType::Concrete(Dtype::U32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: true,
    });
    let counts_chan = p.channels.len() as u32;
    p.channels.push(ChannelDecl {
        shape: v,
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: true,
    });
    let r = p.op(Op::ChanTake(rng_chan));
    let counts = p.op(Op::ChanTake(counts_chan));
    let raw = p.op(Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, vocab),
        dtype: Dtype::F32,
    });
    let logits = p.op(Op::Reshape {
        value: raw,
        shape: v,
    });
    let fp = p.f(0.7);
    let fpb = p.op(Op::Broadcast {
        value: fp,
        shape: v,
    });
    let freq = p.op(Op::Mul(fpb, counts));
    let logits = p.op(Op::Sub(logits, freq));
    let noise = p.op(Op::RngKeyed {
        state: r,
        shape: v,
        kind: RngKind::Gumbel,
    });
    let perturbed = p.op(Op::Add(logits, noise));
    let token = p.op(Op::ReduceArgmax(perturbed));
    let token1 = p.op(Op::Reshape {
        value: token,
        shape: Shape::vector(1),
    });
    let one = p.f(1.0);
    let next = p.op(Op::ScatterAdd {
        base: counts,
        idx: token,
        vals: one,
    });
    let step = p.op(Op::Iota { len: 2 });
    let r_next = p.op(Op::Add(r, step));
    p.op(Op::ChanPut {
        chan: rng_chan,
        value: r_next,
    });
    p.op(Op::ChanPut {
        chan: counts_chan,
        value: next,
    });
    let out = p.output(Shape::vector(1), Dtype::I32, token1);
    (p.container(), out)
}

fn carried_cells_answer_what_the_rings_answer(device: &Device, stages: &mut Stages) {
    let (container, out) = carrying(VOCAB);
    let plan = plan_of(container, VOCAB);
    let carried = engine_xla::guest::carried_channels(&plan.package);
    assert_eq!(
        carried,
        vec![0, 1],
        "the rng and the histogram stay on the device"
    );
    let lanes = 3u32;
    let key = key_of(&plan);
    let seeds: Vec<Vec<(u32, Value)>> = (0..lanes)
        .map(|i| {
            vec![
                (0, Value::U32(vec![9 + i, 1])),
                (1, Value::F32(vec![0.0; VOCAB as usize])),
            ]
        })
        .collect();
    let mut device_insts: Vec<InterpInstance> = seeds.iter().map(|s| instance(&plan, s)).collect();
    let mut host_insts: Vec<InterpInstance> = seeds.iter().map(|s| instance(&plan, s)).collect();
    // A small pool: each lane's cells in rows of their own, the sink last.
    let words = engine_xla::guest::carried_words(&plan.package);
    let per_lane: u32 = words
        .iter()
        .map(|&(_, w)| (engine_xla::guest::lower::cell_rows(w as usize) as u32).next_multiple_of(2))
        .sum();
    let pool_rows = per_lane * lanes + 8;
    let sink_row = pool_rows - 8;
    let cells: Vec<Vec<(u32, u32)>> = (0..lanes)
        .map(|l| {
            let mut at = l * per_lane;
            words
                .iter()
                .map(|&(c, w)| {
                    let first = at;
                    at += (engine_xla::guest::lower::cell_rows(w as usize) as u32).next_multiple_of(2);
                    (c, first)
                })
                .collect()
        })
        .collect();
    stages.pool_rows = i64::from(pool_rows);
    stages.sink_row = sink_row;
    let mut pool = device
        .zeros_or_dry(dtype::Dtype::U32, pool_rows, 128)
        .expect("the pool");
    {
        let writes: Vec<(&InterpInstance, &[(u32, u32)])> = device_insts
            .iter()
            .zip(&cells)
            .map(|(i, c)| (i, c.as_slice()))
            .collect();
        engine_xla::guest::run::write_cells(
            device,
            &plan,
            &mut pool,
            i64::from(pool_rows),
            sink_row,
            &writes,
        )
        .expect("the seeds land in the slots");
    }
    for pass in 0..4 {
        let plane = logits_plane(lanes, VOCAB, 70 + pass);
        let kept = kept(device, &plane, lanes, (0..lanes).map(|l| (l, 1)).collect());
        let mut members: Vec<Member<'_>> = device_insts
            .iter_mut()
            .enumerate()
            .map(|(l, inst)| Member {
                inst,
                bound: bound(&plan, &kept, l),
                cells: cells[l].clone(),
            })
            .collect();
        let outcomes = step_group(
            device,
            stages,
            key,
            &plan,
            Some(&kept),
            &mut members,
            &carried,
            &mut pool,
            eta_ir::registry::Stage::Epilogue,
        );
        assert!(
            outcomes.iter().all(|o| *o == StepOutcome::Committed),
            "{outcomes:?}"
        );
        for (l, inst) in host_insts.iter_mut().enumerate() {
            let row = &plane[l * VOCAB as usize..(l + 1) * VOCAB as usize];
            let outputs = [("logits".to_string(), 0, Value::F32(row.to_vec()))];
            let inputs = PassInputs { outputs: &outputs };
            assert_eq!(eta_exec::step(inst, &plan, &inputs), StepOutcome::Committed);
            let want = take_all(&plan, inst, &[out]);
            let got = take_all(&plan, &device_insts[l], &[out]);
            exact(&got[0], &want[0], &format!("pass {pass} lane {l} token"));
        }
    }
    // Back to the rings: the cells the device carried are the interpreter's.
    let whole = pool.as_ref().expect("the pool");
    for (inst, c) in device_insts.iter().zip(&cells) {
        engine_xla::guest::run::read_cells(&plan, whole, c, inst).expect("the cells come back");
    }
    for (d, h) in device_insts.iter().zip(&host_insts) {
        for c in 0..2 {
            exact(
                &d.channels[c].front(),
                &h.channels[c].front(),
                &format!("carried ring {c}"),
            );
        }
    }
}

/// `PIE_XLA_GUEST_BENCH=1`: the sampler over a 64-lane group, its carried
/// cells in the pool, timed over 50 passes after 3 that warm.
#[test]
fn a_guest_group_pass_is_timed() {
    if std::env::var_os("PIE_XLA_GUEST_BENCH").is_none() {
        return;
    }
    let Some(dev) = device() else {
        return;
    };
    let device = &dev.device;
    let vocab: u32 = std::env::var("PIE_XLA_GUEST_BENCH_VOCAB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32768);
    let lanes: u32 = std::env::var("PIE_XLA_GUEST_BENCH_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let (container, out) = carrying(vocab);
    let outs = [out];
    let plan = plan_of(container, vocab);
    let carried = engine_xla::guest::carried_channels(&plan.package);
    let key = key_of(&plan);
    let words = engine_xla::guest::carried_words(&plan.package);
    let per_lane: u32 = words
        .iter()
        .map(|&(_, w)| (engine_xla::guest::lower::cell_rows(w as usize) as u32).next_multiple_of(2))
        .sum();
    let pool_rows = per_lane * lanes + 8;
    let mut stages = Stages::default();
    stages.top = lanes;
    stages.pool_rows = i64::from(pool_rows);
    stages.sink_row = pool_rows - 8;
    let seeds = |i: u32| {
        vec![
            (0, Value::U32(vec![9 + i, 1])),
            (1, Value::F32(vec![0.0; vocab as usize])),
        ]
    };
    let mut insts: Vec<InterpInstance> = (0..lanes).map(|i| instance(&plan, &seeds(i))).collect();
    let cells: Vec<Vec<(u32, u32)>> = (0..lanes)
        .map(|l| {
            let mut at = l * per_lane;
            words
                .iter()
                .map(|&(c, w)| {
                    let first = at;
                    at += (engine_xla::guest::lower::cell_rows(w as usize) as u32).next_multiple_of(2);
                    (c, first)
                })
                .collect()
        })
        .collect();
    let mut pool = device
        .zeros_or_dry(dtype::Dtype::U32, pool_rows, 128)
        .expect("the pool");
    {
        let writes: Vec<(&InterpInstance, &[(u32, u32)])> =
            insts.iter().zip(&cells).map(|(i, c)| (i, c.as_slice())).collect();
        engine_xla::guest::run::write_cells(
            device, &plan, &mut pool, i64::from(pool_rows), stages.sink_row, &writes,
        )
        .expect("the seeds land");
    }
    let plane = logits_plane(lanes, vocab, 5);
    let kept = kept(device, &plane, lanes, (0..lanes).map(|l| (l, 1)).collect());
    let mut timed = std::time::Duration::ZERO;
    for pass in 0..53 {
        // Fresh rings each pass (the inputs refilled); the carried cells
        // stay in the pool.
        insts = (0..lanes).map(|i| instance(&plan, &seeds(i))).collect();
        let mut members: Vec<Member<'_>> = insts
            .iter_mut()
            .enumerate()
            .map(|(l, inst)| Member {
                inst,
                bound: bound(&plan, &kept, l),
                cells: cells[l].clone(),
            })
            .collect();
        let t = std::time::Instant::now();
        let outcomes = step_group(
            device, &mut stages, key, &plan, Some(&kept), &mut members, &carried, &mut pool,
            eta_ir::registry::Stage::Epilogue,
        );
        if pass >= 3 {
            timed += t.elapsed();
        }
        assert!(outcomes.iter().all(|o| *o == StepOutcome::Committed), "{outcomes:?}");
        for inst in &insts {
            let _ = take_all(&plan, inst, &outs);
        }
    }
    eprintln!(
        "GUEST BENCH carrying vocab {vocab} lanes {lanes} carried {:?}: {:.3} ms a pass",
        carried,
        timed.as_secs_f64() * 1e3 / 50.0
    );
}

/// `PIE_XLA_GUEST_BENCH=1`: a serving step's shape, a model fire then a
/// guest pass, alternating: 64 decode lanes of qwen3-30b-a3b-mini, then the
/// carrying program over 64 lanes (vocab 32768 cells) in the model's own
/// pool. Timed over 50 steps after 3 that warm.
#[test]
fn a_fire_then_a_guest_pass_is_timed() {
    use crate::common_llm::{Model, Refs, Seq, tokens};
    use engine_xla::store::layout::CHANNEL;
    if std::env::var_os("PIE_XLA_GUEST_BENCH").is_none() {
        return;
    }
    if device().is_none() {
        return;
    }
    let (vocab, lanes) = (32768u32, 64u32);
    let mut model = Model::load_with("qwen3-30b-a3b-mini", None, 256, 7, None);
    let mut refs = Refs::of(&model);
    let mvocab = model.vocab;
    let mut seqs: Vec<Seq> = (0..lanes).map(Seq::new).collect();
    let prompts: Vec<_> = seqs
        .iter_mut()
        .enumerate()
        .map(|(i, s)| s.lane(&model, &mut refs, tokens(i as u64, 16, mvocab)))
        .collect();
    model.fire(prompts);
    for s in &mut seqs {
        s.held = 16;
    }
    // Each lane's cells: whole Channel slots, contiguous rows.
    let (container, out) = carrying(vocab);
    let plan = plan_of(container, vocab);
    let carried = engine_xla::guest::carried_channels(&plan.package);
    let key = key_of(&plan);
    let words = engine_xla::guest::carried_words(&plan.package);
    let rows_of = |w: u32| (engine_xla::guest::lower::cell_rows(w as usize) as u32).next_multiple_of(2);
    let per_lane: u32 = words.iter().map(|&(_, w)| rows_of(w)).sum();
    let mut free: Vec<u32> = Vec::new();
    while (free.len() as u32) * 2 < per_lane * lanes + 4096 {
        let slot = refs.take(CHANNEL);
        let row = model.engine.shell().expect("loaded").channel_row(slot).expect("a row");
        free.push(row);
    }
    free.sort_unstable();
    let mut cells: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut at = 0usize;
    while cells.len() < lanes as usize {
        // A run of contiguous slots long enough for one lane.
        let need = (per_lane / 2) as usize;
        let run_ok = (0..need).all(|k| free.get(at + k) == Some(&(free[at] + 2 * k as u32)));
        if !run_ok {
            at += 1;
            continue;
        }
        let mut row = free[at];
        cells.push(
            words
                .iter()
                .map(|&(c, w)| {
                    let first = row;
                    row += rows_of(w);
                    (c, first)
                })
                .collect(),
        );
        at += need;
    }
    let mut stages = Stages::default();
    stages.top = lanes;
    {
        let shell = model.engine.shell().expect("loaded");
        let (rows, sink) = shell.pool_geometry();
        stages.pool_rows = rows;
        stages.sink_row = sink;
    }
    let seeds = |i: u32| {
        vec![
            (0, Value::U32(vec![9 + i, 1])),
            (1, Value::F32(vec![0.0; vocab as usize])),
        ]
    };
    let mut insts: Vec<InterpInstance> = (0..lanes).map(|i| instance(&plan, &seeds(i))).collect();
    {
        let shell = model.engine.shell_mut().expect("loaded");
        let (device, pool) = shell.device_and_pool();
        let writes: Vec<(&InterpInstance, &[(u32, u32)])> =
            insts.iter().zip(&cells).map(|(i, c)| (i, c.as_slice())).collect();
        engine_xla::guest::run::write_cells(
            device, &plan, pool, stages.pool_rows, stages.sink_row, &writes,
        )
        .expect("the seeds land");
    }
    let plane = logits_plane(lanes, vocab, 5);
    let kept = {
        let shell = model.engine.shell().expect("loaded");
        kept(shell.device(), &plane, lanes, (0..lanes).map(|l| (l, 1)).collect())
    };
    let mut timed = std::time::Duration::ZERO;
    for step in 0..53u64 {
        let t = std::time::Instant::now();
        let lanes_now: Vec<_> = seqs
            .iter_mut()
            .enumerate()
            .map(|(i, s)| s.lane(&model, &mut refs, tokens(100 + step * 64 + i as u64, 1, mvocab)))
            .collect();
        model.fire(lanes_now);
        for s in &mut seqs {
            s.held += 1;
        }
        insts = (0..lanes).map(|i| instance(&plan, &seeds(i))).collect();
        {
            let shell = model.engine.shell_mut().expect("loaded");
            let (device, pool) = shell.device_and_pool();
            let mut members: Vec<Member<'_>> = insts
                .iter_mut()
                .enumerate()
                .map(|(l, inst)| Member {
                    inst,
                    bound: bound(&plan, &kept, l),
                    cells: cells[l].clone(),
                })
                .collect();
            let outcomes = step_group(
                device, &mut stages, key, &plan, Some(&kept), &mut members, &carried, pool,
                eta_ir::registry::Stage::Epilogue,
            );
            assert!(outcomes.iter().all(|o| *o == StepOutcome::Committed), "{outcomes:?}");
        }
        if step >= 3 {
            timed += t.elapsed();
        }
        for inst in &insts {
            let _ = take_all(&plan, inst, &[out]);
        }
    }
    eprintln!(
        "STEP BENCH fire+pass, 64 lanes, cells vocab {vocab}: {:.3} ms a step",
        timed.as_secs_f64() * 1e3 / 50.0
    );
}

/// `PIE_XLA_GUEST_BENCH=1`: a sampler registers on a served model, its
/// ladder sealed (qwen3-30b-a3b-mini at `PIE_XLA_GUEST_LANES` lanes, 256 unset, prewarmed):
/// how many stage programs registration compiles and how long it takes,
/// none of them a ladder miss.
#[test]
fn a_sampler_registers_on_a_served_model() {
    use crate::common_llm::{Model, set_knob};
    use engine::Engine as _;
    if std::env::var_os("PIE_XLA_GUEST_BENCH").is_none() {
        return;
    }
    if device().is_none() {
        return;
    }
    let lanes = std::env::var("PIE_XLA_GUEST_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256u32);
    set_knob("SLOTS", lanes);
    set_knob("LANES", lanes);
    set_knob("TOKENS", 2 * lanes);
    let mut model = Model::load_with("qwen3-30b-a3b-mini", None, 512, 7, None);
    let device = model.engine.shell().expect("loaded").device();
    let (misses, held) = (device.misses(), device.compiled());
    let reads = model.engine.shell().expect("loaded").guest_planes();
    let (container, _) = carrying(model.vocab);
    let plan = plan_of(container, model.vocab);
    let t = std::time::Instant::now();
    model
        .engine
        .register_program(&engine::program::ProgramRegistration {
            launch: plan.package.clone(),
            ..Default::default()
        })
        .expect("the sampler registers");
    let took = t.elapsed();
    let device = model.engine.shell().expect("loaded").device();
    eprintln!(
        "REGISTRATION {} stage program(s) in {:.1} s over {} plane read(s)",
        device.compiled() - held,
        took.as_secs_f64(),
        reads.len()
    );
    assert_eq!(device.misses(), misses, "registration compiles are not ladder misses");
}
