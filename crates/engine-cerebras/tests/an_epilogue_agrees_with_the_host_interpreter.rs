//! Every guest stage the device runs answers what the host interpreter
//! (`eta_exec::step`) answers on the same inputs: bit for bit where the
//! interpreter is exact (integers, orderings and their ties, masks, argmax,
//! the uniform draw), within rounding for float math.
//!
//! Each program is stepped twice from the same seeds: once by the
//! interpreter over a host logits plane, once with every stage lowered to a
//! CSL phase program and run on the fabric simulator, reading the same plane
//! as the fire's kept readout.

use std::collections::BTreeMap;

use engine_cerebras::device::{Buffer, Device, Platform};
use engine_cerebras::guest::run::{OnDevice, Stages};
use engine_cerebras::readout::{Kept, Seat};
use engine_cerebras::sdk::Target;
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
    if let Err(why) = engine_cerebras::guest::admits(&plan.package) {
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

fn device() -> Option<Device> {
    match Device::open(Platform::Simulator {
        target: Target::Wse3,
    }) {
        Ok(device) => Some(device),
        Err(e) => {
            eprintln!("no simulator ({e}): skipping");
            None
        }
    }
}

fn kept(_device: &Device, plane: &[f32], rows: u32, layout: Vec<(u32, u32)>) -> Kept {
    kept_plane(plane, rows, layout)
}

fn kept_plane(plane: &[f32], rows: u32, layout: Vec<(u32, u32)>) -> Kept {
    let width = plane.len() as u32 / rows.max(1);
    let words: Vec<u32> = plane.iter().map(|x| x.to_bits()).collect();
    let buffer = Buffer::new(dtype::Dtype::F32, rows, width, words).expect("an f32 plane");
    Kept::new(buffer, rows, width, None, layout)
}

/// PEs a lane spreads over when a case asks for a spread other than the
/// fewest that fit (0: the fewest).
static FORCE_COLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn spread_over(cols: u32) {
    FORCE_COLS.store(cols, std::sync::atomic::Ordering::Relaxed);
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
    use engine_cerebras::guest::run::{Unfit, columns_above, columns_of, prepare};
    let mut inst = instance(plan, seeds);
    let mut cols = match FORCE_COLS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => columns_of(&plan.package).expect("a spread fits"),
        forced => forced,
    };
    // As the engine does: a stage the linker finds too big widens the
    // spread by a quarter and tries again.
    loop {
        match prepare(device, stages, &plan.package, key_of(plan), cols, 1, kept) {
            Ok(()) => break,
            Err(Unfit::Memory(why)) => {
                let wider = columns_above(&plan.package, cols + cols / 4)
                    .unwrap_or_else(|| panic!("no wider spread fits after {cols}: {why}"));
                eprintln!("{cols} PEs overflow ({why:.80}...); widening to {wider}");
                cols = wider;
            }
            Err(Unfit::Other(why)) => panic!("the stages do not compile: {why}"),
        }
    }
    let mut runner = OnDevice::new(device, stages, key_of(plan), cols, kept, seat);
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

const ROWS: u32 = 4;
const VOCAB: u32 = 48;

/// engine-cuda's epilogue: softmax, gather_row, argmax, top-k, row sums.
fn statistics(vocab: u32) -> (TraceContainer, Vec<u32>) {
    let mut p = Program::new();
    let (_, temp) = p.input(Shape::vector(1), Dtype::F32);
    let temp = p.op(Op::Reshape {
        value: temp,
        shape: Shape::new(&[]).unwrap(),
    });
    let (_, index) = p.input(Shape::vector(ROWS), Dtype::I32);
    let logits = p.op(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(ROWS, vocab),
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
        shape: Shape::matrix(ROWS, vocab),
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
        shape: Shape::matrix(ROWS, vocab),
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
    let k = p.op(Op::Const(Literal::U32(12)));
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

/// Softmax rows and one pivot over them: `which` 0 rank_le(k), 1
/// cummass_le(p), 2 prob_ge(t); the mask and the row sums out.
fn pivot_only(rows: u32, vocab: u32, which: u8) -> (TraceContainer, Vec<u32>) {
    let grid = Shape::matrix(rows, vocab);
    let mut p = Program::new();
    let (_, k) = p.input(Shape::vector(rows), Dtype::I32);
    let (_, top_p) = p.input(Shape::vector(rows), Dtype::F32);
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
    let predicate = match which {
        0 => Predicate::RankLe(k),
        1 => Predicate::CummassLe(top_p),
        _ => Predicate::ProbGe(top_p),
    };
    let keep = p.op(Op::PivotThreshold {
        input: probs,
        predicate,
    });
    let outs = vec![
        p.output(grid, Dtype::Bool, keep),
        p.output(Shape::vector(rows), Dtype::F32, s),
        p.output(grid, Dtype::F32, probs),
    ];
    (p.container(), outs)
}

fn the_pivot_agrees(device: &Device, stages: &mut Stages, rows: u32, vocab: u32, which: u8) {
    let swapped = std::env::var_os("PIE_GUEST_SWAP").is_some();
    let (container, outs) = pivot_only(rows, vocab, which);
    let plan = plan_of(container, vocab);
    let seeds = vec![
        (
            0,
            Value::I32(if swapped {
                vec![20, 1, 0, 5000]
            } else {
                vec![1, 20, 0, 5000]
            }),
        ),
        (
            1,
            Value::F32(if swapped {
                vec![0.95, 0.5, 1.0, 0.0]
            } else {
                vec![0.5, 0.95, 1.0, 0.0]
            }),
        ),
    ];
    let plane = logits_plane(rows, vocab, 11);
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
    close(&got[1], &want[1], 1e-4, "the row sums");
    close(&got[2], &want[2], 1e-5, "the probabilities");
    exact(&got[0], &want[0], "the kept lanes");
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
    // `PIE_GUEST_PLAN=1`: no device, only the spreads the wide cases pick
    // and the words their stages hold at a few spreads.
    if std::env::var_os("PIE_GUEST_PLAN").is_some() {
        use engine_cerebras::guest::lower::{Batch, lower};
        let wide = 4096;
        let plans = [
            ("statistics", plan_of(statistics(wide).0, wide)),
            ("sampler", plan_of(sampler(wide).0, wide)),
            ("rowwise", plan_of(rowwise(4, wide).0, wide)),
        ];
        for (name, plan) in &plans {
            let cols = engine_cerebras::guest::run::columns_of(&plan.package);
            eprintln!("{name}: columns_of = {cols:?}");
            for cols in [8u32, 16, 24, 32, 52, 64, 66, 128] {
                let words: Vec<String> = (0..plan.package.stages.len())
                    .map(|at| {
                        let logits = std::env::var("PIE_GUEST_PLAN").ok().and_then(|v| v.parse::<u32>().ok()).filter(|w| *w > 1);
                        match lower(&plan.package, at, Batch { lanes: 1, cols, logits, mtp: None }) {
                            Ok(l) => l.words.to_string(),
                            Err(why) => format!("refused({})", &why.0[..why.0.len().min(90)]),
                        }
                    })
                    .collect();
                eprintln!("  {cols:>3} cols: {}", words.join(" | "));
            }
            // As the test's widening does: the spread above 65 that lowers,
            // then the batch `prepare` lowers with (the kept readout).
            if *name == "rowwise" {
                let rows = 4;
                let plane = logits_plane(rows, wide, 11);
                let kept = kept_plane(&plane, rows, vec![(0, rows)]);
                let above = engine_cerebras::guest::run::columns_above(&plan.package, 65);
                eprintln!("  columns_above(65) = {above:?}");
                if let Some(cols) = above {
                    for at in 0..plan.package.stages.len() {
                        let batch = engine_cerebras::guest::run::batch_of(&plan.package, at, 1, cols, Some(&kept));
                        let got = lower(&plan.package, at, batch).map(|l| l.words).map_err(|w| w.0);
                        eprintln!("  prepare-like stage {at} batch {batch:?}: {got:?}");
                    }
                }
            }
        }
        // The ragged cases at their forced spreads.
        let ragged = [
            ("statistics 4100 over 32", plan_of(statistics(4100).0, 4100), 32u32),
            ("sampler 100 over 2", plan_of(sampler(100).0, 100), 2),
            ("rowwise 4x100 over 2", plan_of(rowwise(4, 100).0, 100), 2),
        ];
        for (name, plan, cols) in &ragged {
            let words: Vec<String> = (0..plan.package.stages.len())
                .map(|at| match lower(&plan.package, at, Batch { lanes: 1, cols: *cols, logits: None, mtp: None }) {
                    Ok(l) => l.words.to_string(),
                    Err(why) => format!("refused({})", &why.0[..why.0.len().min(120)]),
                })
                .collect();
            eprintln!("{name}: {}", words.join(" | "));
        }
        return;
    }
    let Some(device) = device() else {
        return;
    };
    let device = &device;
    let mut stages = Stages::default();

    // `PIE_GUEST_CASES` picks a subset: `narrow` (one PE per lane), `wide`
    // (a lane over a row of PEs), or a program's name.
    let pick = std::env::var("PIE_GUEST_CASES").unwrap_or_default();
    let want = |case: &str, group: &str| pick.is_empty() || pick == group || pick == case;
    // One PE per lane.
    if want("statistics", "narrow") {
        the_statistics_agree(device, &mut stages, VOCAB);
    }
    if want("acceptance", "narrow") {
        the_acceptance_rule_agrees(device, &mut stages);
    }
    if want("corners", "narrow") {
        the_corners_agree(device, &mut stages);
    }
    if want("sampler", "narrow") {
        the_sampler_agrees(device, &mut stages, 64);
    }
    if want("rowwise", "narrow") {
        the_rowwise_sampler_agrees(device, &mut stages, VOCAB);
    }
    // A lane over a row of PEs: a vocabulary no PE holds alone.
    let wide = 4096;
    let (container, _) = statistics(wide);
    let plan = plan_of(container, wide);
    let cols = engine_cerebras::guest::run::columns_of(&plan.package).expect("fits");
    assert!(cols > 1, "a {wide}-wide lane spreads over PEs ({cols})");
    eprintln!("a {wide}-wide statistics lane spreads over {cols} PEs");
    let t0 = std::time::Instant::now();
    if want("wide-statistics", "wide") {
        the_statistics_agree(device, &mut stages, wide);
        eprintln!("wide statistics: {:.0}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    if want("wide-sampler", "wide") {
        the_sampler_agrees(device, &mut stages, wide);
        eprintln!("wide sampler: {:.0}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    if want("wide-rowwise", "wide") {
        the_rowwise_sampler_agrees(device, &mut stages, wide);
        eprintln!("wide rowwise: {:.0}s", t2.elapsed().as_secs_f64());
    }
    // Ragged spreads: an axis the PEs do not divide, so the last PEs hold
    // a partial block or none (4100 over 32 PEs: blocks of 160, PE 25 holds
    // 100, PEs 26..31 none; 100 over 2 PEs: blocks of 64, PE 1 holds 36).
    if want("ragged-statistics", "ragged") {
        spread_over(32);
        let t = std::time::Instant::now();
        the_statistics_agree(device, &mut stages, 4100);
        eprintln!("ragged statistics: {:.0}s", t.elapsed().as_secs_f64());
    }
    if want("ragged-sampler", "ragged") {
        spread_over(2);
        the_sampler_agrees(device, &mut stages, 100);
    }
    if want("ragged-rowwise", "ragged") {
        spread_over(2);
        the_rowwise_sampler_agrees(device, &mut stages, 100);
    }
    // One pivot at a time over ragged rows (`ragged-rank`, `ragged-mass`,
    // `ragged-floor`), for pinning a disagreement down.
    for (name, which) in [
        ("ragged-rank", 0u8),
        ("ragged-mass", 1),
        ("ragged-floor", 2),
    ] {
        if want(name, "pivots") {
            spread_over(2);
            the_pivot_agrees(device, &mut stages, 4, 100, which);
            eprintln!("{name}: agrees");
        }
    }
    // The same over an axis the PEs divide (128 over 2), for telling a
    // ragged fault from a many-rows one.
    for (name, which) in [("even-rank", 0u8), ("even-mass", 1)] {
        if want(name, "even") {
            spread_over(2);
            the_pivot_agrees(device, &mut stages, 4, 128, which);
            eprintln!("{name}: agrees");
        }
    }
    // The sampler at Qwen's vocabulary, on the simulator (hours), at the
    // spread of its own (the pivot cases above forced two PEs; forced over
    // two, a 151936-wide lane is 925k words a PE and the stages refuse).
    spread_over(0);
    if want("qwen-run", "qwen") {
        let t = std::time::Instant::now();
        the_sampler_agrees(device, &mut stages, 151_936);
        eprintln!("qwen sampler: {:.0}s", t.elapsed().as_secs_f64());
    }
    spread_over(0);
    eprintln!("ran {} stage(s) on the simulator", stages.runs);
}

fn the_statistics_agree(device: &Device, stages: &mut Stages, vocab: u32) {
    let (container, outs) = statistics(vocab);
    let plan = plan_of(container, vocab);
    let seeds = vec![
        (0, Value::F32(vec![0.7])),
        (
            1,
            Value::I32(
                (0..ROWS as i32)
                    .map(|r| (r * 131 + 17) % vocab as i32)
                    .collect(),
            ),
        ),
    ];
    let plane = logits_plane(ROWS, vocab, 7);
    let want = host(&plan, &seeds, &plane, ROWS, &outs);
    // The lane's rows sit after two other lanes' in the readout.
    let mut shared = logits_plane(2, vocab, 99);
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
        (0, Value::F32(vec![0.9, 0.05, 2.5, 0.4])),
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
    // One pass at a real vocabulary (`PIE_GUEST_SEEDS=1`), three otherwise.
    let seeds: Vec<u64> = match std::env::var("PIE_GUEST_SEEDS").ok().as_deref() {
        Some("1") => vec![3],
        _ => vec![3, 4, 5],
    };
    for seed in seeds {
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

fn the_rowwise_sampler_agrees(device: &Device, stages: &mut Stages, vocab: u32) {
    let rows = 4;
    let (container, outs) = rowwise(rows, vocab);
    let plan = plan_of(container, vocab);
    let seeds = vec![
        (0, Value::U32(vec![17, 2])),
        (1, Value::I32(vec![1, 20, 0, 5000])),
        (2, Value::F32(vec![0.5, 0.95, 1.0, 0.0])),
        (3, Value::F32(vec![1e-4, 1e-3, 0.0, 0.5])),
    ];
    let plane = logits_plane(rows, vocab, 11);
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

/// The sampler at Qwen's vocabulary compiles for the fabric: every stage
/// lowers at the spread `columns_of` picks and links within a PE (the
/// simulator is not run; `PIE_GUEST_CASES=qwen-compile`).
#[test]
fn the_qwen_sampler_compiles() {
    if std::env::var("PIE_GUEST_CASES").ok().as_deref() != Some("qwen-compile") {
        return;
    }
    let Some(device) = device() else {
        return;
    };
    let vocab = 151_936;
    let (container, _, _) = sampler(vocab);
    let plan = plan_of(container, vocab);
    let cols = engine_cerebras::guest::run::columns_of(&plan.package).expect("a spread fits");
    eprintln!("a {vocab}-wide sampler lane spreads over {cols} PEs");
    let mut stages = Stages::default();
    let t0 = std::time::Instant::now();
    for at in 0..plan.package.stages.len() {
        let batch = engine_cerebras::guest::run::batch_of(&plan.package, at, 1, cols, None);
        let batch = engine_cerebras::guest::lower::Batch {
            logits: Some(vocab),
            ..batch
        };
        let staged = stages
            .get(&device, &plan.package, key_of(&plan), at, batch)
            .unwrap_or_else(|why| panic!("stage {at} does not compile: {why}"));
        eprintln!(
            "stage {at}: {} PEs, compiled in {:.0}s",
            staged.pes,
            t0.elapsed().as_secs_f64()
        );
    }
}

/// The per-PE footprint of every program at each spread, for sizing the
/// guest budget (no device needed).
#[test]
fn footprints_of_the_programs() {
    use engine_cerebras::guest::lower::{Batch, column_choices, lower, wide_axis};
    let named: Vec<(&str, TraceContainer, u32)> = vec![
        ("statistics 48", statistics(VOCAB).0, VOCAB),
        ("acceptance", acceptance().0, VOCAB),
        ("corners", corners().0, VOCAB),
        ("sampler 64", sampler(64).0, 64),
        ("rowwise 48", rowwise(4, VOCAB).0, VOCAB),
        ("statistics 4096", statistics(4096).0, 4096),
        ("sampler 4096", sampler(4096).0, 4096),
        ("rowwise 4096", rowwise(4, 4096).0, 4096),
        // Qwen's vocabulary: what a real lane needs.
        ("statistics 151936", statistics(151_936).0, 151_936),
        ("sampler 151936", sampler(151_936).0, 151_936),
    ];
    for (name, container, vocab) in named {
        let profile = ModelProfile {
            vocab,
            ..ModelProfile::dummy()
        };
        let bound = eta_ir::validate::bind(container, profile).expect("binds");
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let plan = eta_exec::adopt_launch_package(launch).expect("adopts");
        let choices = wide_axis(&plan.package).map_or_else(|| vec![1], column_choices);
        let mut refused = 0;
        for cols in choices {
            let batch = Batch {
                lanes: 1,
                cols,
                logits: None,
                mtp: None,
            };
            match lower(&plan.package, 0, batch) {
                Ok(l) => {
                    eprintln!(
                        "{name}: cols {cols}: {} words, {} segments",
                        l.words,
                        l.text.matches("fn seg").count()
                    );
                    if let Some(dir) = std::env::var_os("PIE_DUMP_GUEST") {
                        let file = format!("{}-{cols}.csl.txt", name.replace(' ', "_"));
                        let _ = std::fs::write(std::path::Path::new(&dir).join(file), &l.text);
                    }
                }
                Err(why) => {
                    refused += 1;
                    if refused <= 2 {
                        eprintln!("{name}: cols {cols}: refused: {why}");
                    }
                }
            }
        }
        if refused > 2 {
            eprintln!("{name}: {} more spreads refused", refused - 2);
        }
    }
}

// ------------------------------------------------------- several lanes

/// Runs one lane of a lockstep batch: every lane deposits its values at
/// each stage, lane 0 runs the stage for all of them in one program (a
/// `cols × lanes` rectangle, PE row `y` lane `y`), and each lane takes its
/// puts back.
struct Lockstep<'a> {
    me: usize,
    cols: u32,
    key: [u8; 32],
    kept: &'a Kept,
    shared: &'a Shared<'a>,
}

struct Shared<'a> {
    device: &'a Device,
    stages: std::sync::Mutex<Stages>,
    barrier: std::sync::Barrier,
    vals: std::sync::Mutex<Vec<Vec<Value>>>,
    outs: std::sync::Mutex<Vec<Vec<(u32, Value)>>>,
    ran: std::sync::atomic::AtomicUsize,
}

impl eta_exec::StageRunner for Lockstep<'_> {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &eta_exec::StagePlan,
        _roots: &[u32],
        _wanted: &[u32],
        vals: &mut [Value],
    ) -> eta_exec::Result<()> {
        use engine_cerebras::guest::run::{Feeding, batch_of, run_stage};
        let at = sp.stage_index;
        let lanes = self.shared.vals.lock().unwrap().len();
        self.shared.vals.lock().unwrap()[self.me] = vals.to_vec();
        self.shared.barrier.wait();
        if self.me == 0 {
            let batch = batch_of(&plan.package, at, lanes as u32, self.cols, Some(self.kept));
            let mut stages = self.shared.stages.lock().unwrap();
            let staged = stages
                .get(self.shared.device, &plan.package, self.key, at, batch)
                .expect("the stage lowers and compiles");
            assert_eq!(staged.pes, self.cols * lanes as u32, "one PE row per lane");
            let all = self.shared.vals.lock().unwrap();
            let feedings: Vec<Feeding<'_>> = (0..lanes)
                .map(|l| Feeding {
                    vals: &all[l],
                    seat: self.kept.seat(l, 1),
                })
                .collect();
            let outs = run_stage(self.shared.device, &staged, Some(self.kept), &feedings)
                .expect("the stage runs");
            *self.shared.outs.lock().unwrap() = outs;
            self.shared
                .ran
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        self.shared.barrier.wait();
        let mine = std::mem::take(&mut self.shared.outs.lock().unwrap()[self.me]);
        for (id, value) in mine {
            vals[id as usize] = value;
        }
        Ok(())
    }

    fn binds(&self, id: Option<IntrinsicId>) -> bool {
        engine_cerebras::guest::run::device_binds(id, Some(self.kept), self.kept.seat(self.me, 1))
    }
}

/// Three sampler lanes share one program on a 2 × 3 rectangle (each lane
/// over two PEs) and every lane samples what the interpreter samples for
/// it.
#[test]
fn several_lanes_share_one_program() {
    let Some(device) = device() else {
        return;
    };
    let vocab = 64u32;
    let (container, outs, _) = sampler(vocab);
    let plan = plan_of(container, vocab);
    let seeds: Vec<u64> = vec![3, 4, 5];
    let lanes = seeds.len();
    let mut plane = Vec::new();
    for &seed in &seeds {
        plane.extend(logits_plane(1, vocab, seed));
    }
    let kept = kept(
        &device,
        &plane,
        lanes as u32,
        (0..lanes as u32).map(|l| (l, 1)).collect(),
    );
    let want: Vec<Vec<Value>> = seeds
        .iter()
        .map(|&seed| {
            host(
                &plan,
                &sampler_seeds(vocab, seed),
                &logits_plane(1, vocab, seed),
                1,
                &outs,
            )
        })
        .collect();
    let shared = Shared {
        device: &device,
        stages: std::sync::Mutex::new(Stages::default()),
        barrier: std::sync::Barrier::new(lanes),
        vals: std::sync::Mutex::new(vec![Vec::new(); lanes]),
        outs: std::sync::Mutex::new(vec![Vec::new(); lanes]),
        ran: std::sync::atomic::AtomicUsize::new(0),
    };
    let got: Vec<Vec<Value>> = std::thread::scope(|s| {
        let handles: Vec<_> = seeds
            .iter()
            .enumerate()
            .map(|(me, &seed)| {
                let (plan, kept, shared, outs) = (&plan, &kept, &shared, &outs);
                s.spawn(move || {
                    let mut inst = instance(plan, &sampler_seeds(vocab, seed));
                    let mut runner = Lockstep {
                        me,
                        cols: 2,
                        key: key_of(plan),
                        kept,
                        shared,
                    };
                    let outcome =
                        eta_exec::step_with(&mut inst, plan, &PassInputs::none(), &mut runner);
                    assert_eq!(outcome, StepOutcome::Committed, "lane {me} commits");
                    take_all(plan, &inst, outs)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ran = shared.ran.load(std::sync::atomic::Ordering::Acquire);
    assert!(ran > 0, "stages ran on the device");
    eprintln!("{ran} stage(s) ran, each carrying {lanes} lanes");
    for (lane, (got, want)) in got.iter().zip(&want).enumerate() {
        let tag = |what: &str| format!("lane {lane}: {what}");
        exact(&got[0], &want[0], &tag("the sampled token"));
        exact(&got[1], &want[1], &tag("the greedy token"));
        exact(&got[2], &want[2], &tag("the next histogram"));
        exact(&got[3], &want[3], &tag("the next rng state"));
        exact(&got[4], &want[4], &tag("the top-k keep"));
        exact(&got[5], &want[5], &tag("the top-p keep"));
        close(&got[6], &want[6], 1e-5, &tag("the probabilities"));
        close(&got[7], &want[7], 1e-5, &tag("the Gumbel draw"));
    }
}
