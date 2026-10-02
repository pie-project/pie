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
use engine_xla::guest::run::{Member, OnDevice, Stages, step_group};
use engine_xla::readout::{Kept, Seat};
use eta_compiler::plan::compile_bound;
use eta_exec::{ExecPlan, InterpInstance, PassInputs, StepOutcome, Value};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage};
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
            names: Vec::new(),
            channels: self.channels,
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: self.ops,
            }],
            externs: Vec::new(),
        }
    }
}

fn plan_of(container: TraceContainer, vocab: u32) -> ExecPlan {
    let profile = ModelProfile {
        vocab,
        ..ModelProfile::dummy()
    };
    let bound = eta_ir::validate::bind(container, profile).expect("the program binds");
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
    rows: u32,
    channels: &[u32],
) -> Vec<Value> {
    let mut inst = instance(plan, seeds);
    let vocab = if rows == 0 {
        0
    } else {
        (logits.len() / rows as usize) as u32
    };
    let inputs = PassInputs {
        logits: (!logits.is_empty()).then_some(logits),
        rows,
        vocab,
        ..PassInputs::none()
    };
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
    Kept::new(buffer, rows, width, None, layout)
}

/// The device's answer for one lane seated at `seat` of `kept`.
fn on_device(
    device: &Device,
    stages: &mut Stages,
    plan: &ExecPlan,
    seeds: &[(u32, Value)],
    kept: Option<&Kept>,
    seat: Option<Seat>,
    channels: &[u32],
) -> Vec<Value> {
    let mut inst = instance(plan, seeds);
    let mut runner = OnDevice::new(device, stages, key_of(plan), kept, seat);
    let outcome = eta_exec::step_with(&mut inst, plan, &PassInputs::none(), &mut runner);
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
    let logits = p.op(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
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
    let raw = p.op(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
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
    let mut mask = vec![1u8; n];
    let mut words = vec![u32::MAX; n.div_ceil(32)];
    for (i, u) in noise(seed ^ 0x55, 64, 0.0, n as f32)
        .into_iter()
        .enumerate()
    {
        let at = (u as usize).min(n - 1);
        match i % 4 {
            0 => counts[at] += 1.0,
            1 => present[at] = 1.0,
            2 => mask[at] = 0,
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
    let logits = p.op(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
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
        1,
        1,
        4,
        i32::MAX,
        15,
    ];
    let b = vec![2, 2, -1, 0, 3, 3, 0, -2, 4, 1, 1, 17, 1, -5, -1, 15];
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
        1,
        1,
        40,
        12,
        15,
    ];
    let ub = vec![2, 2, 0, 0, 3, 7, 0xFFFF_FFF0, 2, 4, 1, 1, 17, 1, 5, 3, 15];
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
    the_acceptance_rule_agrees(device, &mut stages);
    the_corners_agree(device, &mut stages);
    the_sampler_agrees(device, &mut stages, VOCAB);
    the_rowwise_sampler_agrees(device, &mut stages);
    a_group_answers_what_each_lane_answers_alone(device, &mut stages);
    carried_cells_answer_what_the_rings_answer(device, &mut stages);
    eprintln!("compiled {} stage program(s)", stages.len());
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
    let seat = kept.seat(1, 1);
    let got = on_device(device, stages, &plan, &seeds, Some(&kept), seat, &outs);
    close(&got[0], &want[0], 1e-5, "gather_row of the scaled logits");
    exact(&got[1], &want[1], "argmax per row");
    exact(&got[2], &want[2], "top-4 ids per row");
    close(&got[3], &want[3], 1e-5, "top-4 values");
    close(&got[4], &want[4], 1e-4, "softmax row mass");
    close(&got[5], &want[5], 1e-5, "probability at the argmax");
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
    let got = on_device(device, stages, &plan, &seeds, None, None, &outs);
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
    let got = on_device(device, stages, &plan, &seeds, None, None, &outs);
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
            kept.seat(0, 1),
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
        kept.seat(0, 1),
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
            seat: kept.seat(lane, 1),
        })
        .collect();
    let (outcomes, _) = step_group(
        device,
        stages,
        key,
        &plan,
        Some(&kept),
        &mut members,
        &[],
        None,
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
    let raw = p.op(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
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
    let mut cells = None;
    for pass in 0..4 {
        let plane = logits_plane(lanes, VOCAB, 70 + pass);
        let kept = kept(device, &plane, lanes, (0..lanes).map(|l| (l, 1)).collect());
        let mut members: Vec<Member<'_>> = device_insts
            .iter_mut()
            .enumerate()
            .map(|(l, inst)| Member {
                inst,
                seat: kept.seat(l, 1),
            })
            .collect();
        let (outcomes, next) = step_group(
            device,
            stages,
            key,
            &plan,
            Some(&kept),
            &mut members,
            &carried,
            cells.take(),
        );
        assert!(
            outcomes.iter().all(|o| *o == StepOutcome::Committed),
            "{outcomes:?}"
        );
        assert!(next.is_some(), "the pass carried its cells");
        cells = next;
        for (l, inst) in host_insts.iter_mut().enumerate() {
            let row = &plane[l * VOCAB as usize..(l + 1) * VOCAB as usize];
            let inputs = PassInputs {
                logits: Some(row),
                rows: 1,
                vocab: VOCAB,
                ..PassInputs::none()
            };
            assert_eq!(eta_exec::step(inst, &plan, &inputs), StepOutcome::Committed);
            let want = take_all(&plan, inst, &[out]);
            let got = take_all(&plan, &device_insts[l], &[out]);
            exact(&got[0], &want[0], &format!("pass {pass} lane {l} token"));
        }
    }
    // Back to the rings: the cells the device carried are the interpreter's.
    let cells = cells.expect("cells");
    let insts: Vec<Option<&InterpInstance>> = device_insts.iter().map(Some).collect();
    engine_xla::guest::run::flush(&plan, &cells, &insts).expect("the cells come back");
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
