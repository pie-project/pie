//! Every op of the op table on the device against the host interpreter, the
//! tier-0 reference every backend diffs against. Each op runs in a program of
//! its own over rows of several shapes — one element, a single wide row (a
//! decode lane's sampler), two rows (a guided pair), several rows — fired as
//! 1, 8 and 64 lanes at once, over two seeds. Each op reads its operands
//! twice: straight from a channel, and through an identity map, so that a
//! row stream produces the operand the op reads.
//!
//! The answer is compared bit for bit wherever the contract is bitwise:
//! integer and boolean ops, ordering and indexing, IEEE arithmetic (the
//! device compiles without contraction, with exact division and square
//! root), and reductions, whose order the reference fixes (`canonical_reduce`).
//! Only the transcendentals the device rounds its own way (`exp`, `log`,
//! `sin`, `cos`) are held to a few ulps, and the draws built on them to a
//! few ulps of their scale (a Gumbel draw `-log(-log u)` near zero is a
//! difference of two logs). A NaN agrees with any NaN: IEEE leaves the
//! payload of an invalid operation to the machine, and the host's differs
//! from the device's.
//!
//! Coverage is enforced: every op of `OP_TABLE` must appear in some case,
//! but for `kernel_call` and `sink_call`, which name host kernels the
//! interpreter has no value for.

use std::collections::BTreeSet;

use engine::{ChannelId, InstanceId, ProgramRegistration};
use engine_cuda::device::{Buffer, Context};
use engine_cuda::program::Plane;
use engine_cuda::program::compile::Disk;
use engine_cuda::program::launch::INTRINSIC_STORAGE_F32;
use engine_cuda::program::session::{Fired, Launched};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, OP_TABLE, Op, tags};
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId};
use eta_ir::validate::BoundTrace;

/// Rows by width: one element, a single wide row, a guided pair, a batch.
const SHAPES: &[(u32, u32)] = &[(1, 1), (1, 300), (2, 300), (3, 1000)];
const LANES: &[u32] = &[1, 8, 64];
const SEEDS: &[u64] = &[0x9e37_79b9_7f4a_7c15, 0x0bad_cafe_1234_5678];

/// Ops whose value is a host kernel's: the interpreter has none to diff.
const EXEMPT: &[u8] = &[tags::KERNEL_CALL, tags::SINK_CALL];

#[derive(Clone, Copy, Debug)]
enum Gen {
    /// Signed values on a coarse grid: ties, zeros of both signs.
    Signed,
    /// Strictly positive values.
    Positive,
    /// A softmax row: probabilities summing to about one.
    Probs,
    /// Integers in `-span..span`.
    Int(i32),
    /// Unsigned integers below `n`.
    Below(u32),
    Bool,
    /// Indices into an axis of `n`, a few of them out of range.
    Index(u32),
    /// Fractions of a row's mass, for a top-p cut.
    Mass,
}

#[derive(Clone, Copy, PartialEq)]
enum Check {
    Exact,
    /// Within this many ulps.
    Ulps(u32),
    /// Within this many ulps of the larger of one and the expected value.
    Scaled(u32),
}

struct Output {
    chan: u32,
    dtype: Dtype,
    len: usize,
    check: Check,
    what: String,
}

/// A channel the program reads, seeded from `source` in rows `row` wide.
#[derive(Clone, Copy)]
struct Input {
    chan: u32,
    dtype: Dtype,
    len: usize,
    source: Gen,
    row: u32,
}

struct Case {
    name: String,
    rows: u32,
    width: u32,
    container: TraceContainer,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    logits: bool,
}

struct Prog {
    ops: Vec<Op>,
    next: ValueId,
    channels: Vec<ChannelDecl>,
    inputs: Vec<Input>,
    puts: Vec<(u32, ValueId)>,
    outputs: Vec<Output>,
    logits: bool,
    /// Whether operands reach the op through an identity map.
    mapped: bool,
}

impl Prog {
    fn new(mapped: bool) -> Prog {
        Prog {
            ops: Vec::new(),
            next: 0,
            channels: Vec::new(),
            inputs: Vec::new(),
            puts: Vec::new(),
            outputs: Vec::new(),
            logits: false,
            mapped,
        }
    }

    fn op(&mut self, op: Op) -> ValueId {
        let results = op.result_count();
        let base = self.next;
        self.ops.push(op);
        self.next += results;
        base
    }

    fn input_dims(&mut self, dims: &[u32], dtype: Dtype, source: Gen) -> ValueId {
        self.input_with(dims, dtype, source, false)
    }

    /// A channel `dims` wide seeded from `source` (a row of `dims`' last width
    /// for row generators), read — or taken, with `take` — then mapped.
    fn input_with(&mut self, dims: &[u32], dtype: Dtype, source: Gen, take: bool) -> ValueId {
        let shape = Shape::new(dims).expect("a shape");
        let chan = self.channels.len() as u32;
        self.channels.push(ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::Writer,
            seeded: true,
        });
        let row = dims.last().copied().unwrap_or(1).max(1);
        self.inputs.push(Input {
            chan,
            dtype,
            len: shape.numel() as usize,
            source,
            row,
        });
        let read = self.op(if take {
            Op::ChanTake(chan)
        } else {
            Op::ChanRead(chan)
        });
        if !self.mapped {
            return read;
        }
        match dtype {
            Dtype::Bool => {
                let once = self.op(Op::Not(read));
                self.op(Op::Not(once))
            }
            _ => {
                let once = self.op(Op::Neg(read));
                self.op(Op::Neg(once))
            }
        }
    }

    fn logits(&mut self, rows: u32, width: u32) -> ValueId {
        self.logits = true;
        self.op(Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            lane: 0,
            shape: Shape::matrix(rows, width),
            dtype: Dtype::F32,
        })
    }

    fn output(&mut self, value: ValueId, dims: &[u32], dtype: Dtype, check: Check, what: &str) {
        let shape = Shape::new(dims).expect("a shape");
        let chan = self.channels.len() as u32;
        self.channels.push(ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        });
        self.puts.push((chan, value));
        self.outputs.push(Output {
            chan,
            dtype,
            len: shape.numel() as usize,
            check,
            what: what.into(),
        });
    }

    fn into_case(mut self, name: String, rows: u32, width: u32) -> Case {
        for (chan, value) in std::mem::take(&mut self.puts) {
            self.ops.push(Op::ChanPut { chan, value });
        }
        let container = TraceContainer {
            names: Vec::new(),
            channels: self.channels,
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: self.ops,
            }],
            externs: Vec::new(),
            inputs: Vec::new(),
        };
        Case {
            name,
            rows,
            width,
            container,
            inputs: self.inputs,
            outputs: self.outputs,
            logits: self.logits,
        }
    }
}

fn scalar() -> Shape {
    Shape::new(&[]).unwrap()
}

/// The body of one op's program over `[rows, width]` operands; each case
/// writes its results through `p.output`.
fn body(tag: u8, p: &mut Prog, r: u32, w: u32) {
    let plane = [r, w];
    let f = Dtype::F32;
    let exact = Check::Exact;
    let unary_f32 = |p: &mut Prog, op: fn(ValueId) -> Op, source: Gen, check: Check, what: &str| {
        let x = p.input_dims(&plane, f, source);
        let y = p.op(op(x));
        p.output(y, &plane, f, check, what);
    };
    match tag {
        tags::EXP => unary_f32(p, Op::Exp, Gen::Signed, Check::Ulps(2), "exp"),
        tags::LOG => unary_f32(p, Op::Log, Gen::Positive, Check::Ulps(2), "log"),
        tags::SIN => unary_f32(p, Op::Sin, Gen::Signed, Check::Ulps(2), "sin"),
        tags::COS => unary_f32(p, Op::Cos, Gen::Signed, Check::Ulps(2), "cos"),
        tags::RECIP => unary_f32(p, Op::Recip, Gen::Signed, exact, "recip"),
        tags::SQRT => unary_f32(p, Op::Sqrt, Gen::Positive, exact, "sqrt"),
        tags::RSQRT => unary_f32(p, Op::Rsqrt, Gen::Positive, exact, "rsqrt"),
        tags::NEG | tags::ABS | tags::SIGN => {
            let op = |v| match tag {
                tags::NEG => Op::Neg(v),
                tags::ABS => Op::Abs(v),
                _ => Op::Sign(v),
            };
            for (dtype, source) in [
                (Dtype::F32, Gen::Signed),
                (Dtype::I32, Gen::Int(1000)),
                (Dtype::U32, Gen::Below(1000)),
            ] {
                let x = p.input_dims(&plane, dtype, source);
                let y = p.op(op(x));
                p.output(y, &plane, dtype, exact, &format!("{dtype:?}"));
            }
        }
        tags::CAST => {
            let x = p.input_dims(&plane, f, Gen::Signed);
            let i = p.input_dims(&plane, Dtype::I32, Gen::Int(100_000));
            let u = p.input_dims(&plane, Dtype::U32, Gen::Below(100_000));
            let b = p.input_dims(&plane, Dtype::Bool, Gen::Bool);
            for (value, from, to) in [
                (x, f, Dtype::I32),
                (x, f, Dtype::U32),
                (x, f, Dtype::Bool),
                (i, Dtype::I32, f),
                (i, Dtype::I32, Dtype::U32),
                (i, Dtype::I32, Dtype::Bool),
                (u, Dtype::U32, f),
                (u, Dtype::U32, Dtype::I32),
                (b, Dtype::Bool, f),
                (b, Dtype::Bool, Dtype::I32),
            ] {
                let y = p.op(Op::Cast { value, dtype: to });
                p.output(y, &plane, to, exact, &format!("{from:?} as {to:?}"));
            }
        }
        tags::ADD
        | tags::SUB
        | tags::MUL
        | tags::DIV
        | tags::MAX_ELEM
        | tags::MIN_ELEM
        | tags::REM => {
            let op = |a, b| match tag {
                tags::ADD => Op::Add(a, b),
                tags::SUB => Op::Sub(a, b),
                tags::MUL => Op::Mul(a, b),
                tags::DIV => Op::Div(a, b),
                tags::MAX_ELEM => Op::MaxElem(a, b),
                tags::MIN_ELEM => Op::MinElem(a, b),
                _ => Op::Rem(a, b),
            };
            for (dtype, source) in [
                (Dtype::F32, Gen::Signed),
                (Dtype::I32, Gen::Int(50)),
                (Dtype::U32, Gen::Below(50)),
            ] {
                let a = p.input_with(&plane, dtype, source, true);
                let b = p.input_dims(&plane, dtype, source);
                let y = p.op(op(a, b));
                p.output(y, &plane, dtype, exact, &format!("{dtype:?}"));
                // Against a scalar: a temperature, a constant offset.
                let s = p.input_dims(&[1], dtype, source);
                let s = p.op(Op::Reshape {
                    value: s,
                    shape: scalar(),
                });
                let y = p.op(op(a, s));
                p.output(y, &plane, dtype, exact, &format!("{dtype:?} by a scalar"));
            }
        }
        tags::GT | tags::GE | tags::EQ | tags::NE | tags::LT | tags::LE => {
            let op = |a, b| match tag {
                tags::GT => Op::Gt(a, b),
                tags::GE => Op::Ge(a, b),
                tags::EQ => Op::Eq(a, b),
                tags::NE => Op::Ne(a, b),
                tags::LT => Op::Lt(a, b),
                _ => Op::Le(a, b),
            };
            for (dtype, source) in [
                (Dtype::F32, Gen::Signed),
                (Dtype::I32, Gen::Int(4)),
                (Dtype::U32, Gen::Below(4)),
            ] {
                let a = p.input_dims(&plane, dtype, source);
                let b = p.input_dims(&plane, dtype, source);
                let y = p.op(op(a, b));
                p.output(y, &plane, Dtype::Bool, exact, &format!("{dtype:?}"));
            }
        }
        tags::AND | tags::OR => {
            let a = p.input_dims(&plane, Dtype::Bool, Gen::Bool);
            let b = p.input_dims(&plane, Dtype::Bool, Gen::Bool);
            let y = p.op(if tag == tags::AND {
                Op::And(a, b)
            } else {
                Op::Or(a, b)
            });
            p.output(y, &plane, Dtype::Bool, exact, "bool");
        }
        tags::NOT => {
            let a = p.input_dims(&plane, Dtype::Bool, Gen::Bool);
            let y = p.op(Op::Not(a));
            p.output(y, &plane, Dtype::Bool, exact, "bool");
        }
        tags::SELECT => {
            let c = p.input_dims(&plane, Dtype::Bool, Gen::Bool);
            for (dtype, source) in [(f, Gen::Signed), (Dtype::I32, Gen::Int(100))] {
                let a = p.input_dims(&plane, dtype, source);
                let b = p.input_dims(&plane, dtype, source);
                let y = p.op(Op::Select { cond: c, a, b });
                p.output(y, &plane, dtype, exact, &format!("{dtype:?}"));
            }
        }
        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX => {
            let op = |v| match tag {
                tags::REDUCE_SUM => Op::ReduceSum(v),
                tags::REDUCE_MAX => Op::ReduceMax(v),
                tags::REDUCE_MIN => Op::ReduceMin(v),
                _ => Op::ReduceArgmax(v),
            };
            let out = |dtype| {
                if tag == tags::REDUCE_ARGMAX {
                    Dtype::I32
                } else {
                    dtype
                }
            };
            for (dtype, source) in [
                (f, Gen::Signed),
                (f, Gen::Probs),
                (Dtype::I32, Gen::Int(1000)),
            ] {
                let x = p.input_dims(&plane, dtype, source);
                let y = p.op(op(x));
                p.output(y, &[r], out(dtype), exact, &format!("{dtype:?} {source:?}"));
            }
            // The row the logits are.
            let l = p.logits(r, w);
            let y = p.op(op(l));
            p.output(y, &[r], out(f), exact, "logits");
        }
        tags::BROADCAST => {
            let x = p.input_dims(&[r], f, Gen::Signed);
            let y = p.op(Op::Broadcast {
                value: x,
                shape: Shape::matrix(r, w),
            });
            p.output(y, &plane, f, exact, "a row's scalar across the row");
            let s = p.input_dims(&[1, w], Dtype::I32, Gen::Int(100));
            let y = p.op(Op::Broadcast {
                value: s,
                shape: Shape::matrix(r, w),
            });
            p.output(y, &plane, Dtype::I32, exact, "a shared row to each row");
        }
        tags::RESHAPE => {
            let x = p.input_dims(&plane, f, Gen::Signed);
            let y = p.op(Op::Reshape {
                value: x,
                shape: Shape::vector(r * w),
            });
            p.output(y, &[r * w], f, exact, "flat");
            let y = p.op(Op::Reshape {
                value: x,
                shape: Shape::new(&[r, 1, w]).unwrap(),
            });
            p.output(y, &[r, 1, w], f, exact, "rank 3");
        }
        tags::TRANSPOSE => {
            let x = p.input_dims(&plane, f, Gen::Signed);
            let y = p.op(Op::Transpose(x));
            p.output(y, &[w, r], f, exact, "f32");
            let x = p.input_dims(&plane, Dtype::Bool, Gen::Bool);
            let y = p.op(Op::Transpose(x));
            p.output(y, &[w, r], Dtype::Bool, exact, "bool");
        }
        tags::CUMSUM | tags::CUMPROD => {
            let op = |v| {
                if tag == tags::CUMSUM {
                    Op::CumSum(v)
                } else {
                    Op::CumProd(v)
                }
            };
            for (dtype, source) in [(f, Gen::Probs), (f, Gen::Signed), (Dtype::I32, Gen::Int(3))] {
                let x = p.input_dims(&plane, dtype, source);
                let y = p.op(op(x));
                p.output(y, &plane, dtype, exact, &format!("{dtype:?} {source:?}"));
            }
        }
        tags::SORT_DESC => {
            let n = r * w;
            let x = p.input_dims(&[n], f, Gen::Signed);
            let y = p.op(Op::SortDesc(x));
            p.output(y, &[n], f, exact, "values");
            p.output(y + 1, &[n], Dtype::U32, exact, "order");
        }
        tags::TOP_K => {
            for k in [1, w.min(4), w.min(50)] {
                let x = p.input_dims(&plane, f, Gen::Signed);
                let y = p.op(Op::TopK { input: x, k });
                p.output(y, &[r, k], f, exact, &format!("top-{k} values"));
                p.output(y + 1, &[r, k], Dtype::U32, exact, &format!("top-{k} ids"));
            }
        }
        tags::MATMUL => {
            let n = 8;
            let a = p.input_dims(&plane, f, Gen::Signed);
            let b = p.input_dims(&[w, n], f, Gen::Signed);
            let y = p.op(Op::MatMul(a, b));
            p.output(y, &[r, n], f, exact, "product");
        }
        tags::PIVOT_THRESHOLD => {
            let probs = p.input_dims(&plane, f, Gen::Probs);
            let k = p.input_dims(&[r], Dtype::I32, Gen::Index(w));
            let y = p.op(Op::PivotThreshold {
                input: probs,
                predicate: Predicate::RankLe(k),
            });
            p.output(y, &plane, Dtype::Bool, exact, "rank");
            let mass = p.input_dims(&[r], f, Gen::Mass);
            let y = p.op(Op::PivotThreshold {
                input: probs,
                predicate: Predicate::CummassLe(mass),
            });
            p.output(y, &plane, Dtype::Bool, exact, "mass");
            // A floor the row computes itself: its own peak, scaled.
            let peak = p.op(Op::ReduceMax(probs));
            let half = p.op(Op::Const(Literal::F32(0.5)));
            let floor = p.op(Op::Mul(peak, half));
            let y = p.op(Op::PivotThreshold {
                input: probs,
                predicate: Predicate::ProbGe(floor),
            });
            p.output(y, &plane, Dtype::Bool, exact, "a computed floor");
            let fixed = p.input_dims(&[1], f, Gen::Mass);
            let fixed = p.op(Op::Reshape {
                value: fixed,
                shape: scalar(),
            });
            let y = p.op(Op::PivotThreshold {
                input: probs,
                predicate: Predicate::ProbGe(fixed),
            });
            p.output(y, &plane, Dtype::Bool, exact, "a scalar floor");
        }
        tags::GATHER => {
            let src = p.input_dims(&plane, f, Gen::Signed);
            let idx = p.input_dims(&[r], Dtype::I32, Gen::Index(r));
            let y = p.op(Op::Gather { src, idx });
            p.output(y, &plane, f, exact, "rows by index");
            let flat = p.input_dims(&[w], Dtype::U32, Gen::Below(1000));
            let idx = p.input_dims(&plane, Dtype::I32, Gen::Index(w));
            let y = p.op(Op::Gather { src: flat, idx });
            p.output(y, &plane, Dtype::U32, exact, "elements by index");
        }
        tags::GATHER_ROW => {
            let src = p.input_dims(&plane, f, Gen::Signed);
            let idx = p.input_dims(&[r], Dtype::I32, Gen::Index(w));
            let y = p.op(Op::GatherRow { src, idx });
            p.output(y, &[r], f, exact, "each row's pick");
        }
        tags::SCATTER_ADD | tags::SCATTER_SET => {
            let op = |base, idx, vals| {
                if tag == tags::SCATTER_ADD {
                    Op::ScatterAdd { base, idx, vals }
                } else {
                    Op::ScatterSet { base, idx, vals }
                }
            };
            let base = p.input_dims(&plane, f, Gen::Signed);
            let idx = p.input_dims(&[r], Dtype::I32, Gen::Index(r));
            let vals = p.input_dims(&plane, f, Gen::Signed);
            let y = p.op(op(base, idx, vals));
            p.output(y, &plane, f, exact, "rows");
            let base = p.input_dims(&[w], Dtype::I32, Gen::Int(100));
            let idx = p.input_dims(&[r], Dtype::I32, Gen::Index(w));
            let vals = p.input_dims(&[r], Dtype::I32, Gen::Int(100));
            let y = p.op(op(base, idx, vals));
            p.output(y, &[w], Dtype::I32, exact, "elements");
        }
        tags::IOTA => {
            let y = p.op(Op::Iota { len: w });
            p.output(y, &[w], Dtype::U32, exact, "iota");
            let y = p.op(Op::Reshape {
                value: y,
                shape: Shape::matrix(1, w),
            });
            let y = p.op(Op::Broadcast {
                value: y,
                shape: Shape::matrix(r, w),
            });
            p.output(y, &plane, Dtype::U32, exact, "iota on each row");
        }
        tags::MASK_APPLY_PACKED => {
            let x = p.input_dims(&plane, f, Gen::Signed);
            let words = w.div_ceil(32);
            let mask = p.input_dims(&[words], Dtype::U32, Gen::Below(u32::MAX));
            let y = p.op(Op::MaskApply { logits: x, mask });
            p.output(y, &plane, f, exact, "packed mask");
        }
        tags::CAUSAL_MASK | tags::SLIDING_WINDOW_MASK | tags::SINK_WINDOW_MASK => {
            let positions = p.input_dims(&[r], Dtype::U32, Gen::Below(w + 4));
            let y = p.op(match tag {
                tags::CAUSAL_MASK => Op::CausalMask { positions, len: w },
                tags::SLIDING_WINDOW_MASK => Op::SlidingWindowMask {
                    positions,
                    len: w,
                    window: 3,
                },
                _ => Op::SinkWindowMask {
                    positions,
                    len: w,
                    sink: 2,
                    window: 3,
                },
            });
            p.output(y, &plane, Dtype::Bool, exact, "mask");
        }
        tags::RNG => {
            for (kind, check) in [
                (RngKind::Uniform, exact),
                (RngKind::Gumbel, Check::Scaled(4)),
                (RngKind::Normal, Check::Ulps(4)),
            ] {
                let y = p.op(Op::Rng {
                    stream: 3,
                    shape: Shape::matrix(r, w),
                    kind,
                });
                p.output(y, &plane, f, check, &format!("{kind:?}"));
            }
        }
        tags::RNG_KEYED => {
            let state = p.input_dims(&[2], Dtype::U32, Gen::Below(u32::MAX));
            for (kind, check) in [
                (RngKind::Uniform, exact),
                (RngKind::Gumbel, Check::Scaled(4)),
                (RngKind::Normal, Check::Ulps(4)),
            ] {
                let y = p.op(Op::RngKeyed {
                    state,
                    shape: Shape::matrix(r, w),
                    kind,
                });
                p.output(y, &plane, f, check, &format!("{kind:?}"));
            }
            // A draw of the logits' own shape, added to them: Gumbel-max.
            let l = p.logits(r, w);
            let g = p.op(Op::RngKeyed {
                state,
                shape: Shape::matrix(r, w),
                kind: RngKind::Gumbel,
            });
            let s = p.op(Op::Add(l, g));
            let y = p.op(Op::ReduceArgmax(s));
            p.output(y, &[r], Dtype::I32, exact, "gumbel-max");
        }
        tags::CONST => {
            for (literal, dtype) in [
                (Literal::F32(-0.0), f),
                (Literal::I32(-7), Dtype::I32),
                (Literal::U32(0xdead_beef), Dtype::U32),
                (Literal::Bool(true), Dtype::Bool),
            ] {
                let c = p.op(Op::Const(literal));
                let y = p.op(Op::Broadcast {
                    value: c,
                    shape: Shape::matrix(r, w),
                });
                p.output(y, &plane, dtype, exact, &format!("{dtype:?}"));
            }
        }
        tags::CHAN_TAKE | tags::CHAN_READ | tags::CHAN_PUT => {
            let x = p.input_with(&plane, f, Gen::Signed, tag == tags::CHAN_TAKE);
            p.output(x, &plane, f, exact, "copy");
        }
        tags::INTRINSIC_VAL => {
            let l = p.logits(r, w);
            p.output(l, &plane, f, exact, "logits");
        }
        other => panic!("no case for op tag {other:#x}"),
    }
}

fn case(tag: u8, rows: u32, width: u32) -> Case {
    let name = eta_ir::op::spec(tag).expect("an op table row").name;
    let mut prog = Prog::new(false);
    body(tag, &mut prog, rows, width);
    // The same op again over mapped operands, in the same program.
    let split = prog.outputs.len();
    prog.mapped = true;
    body(tag, &mut prog, rows, width);
    for output in &mut prog.outputs[split..] {
        output.what = format!("{} (mapped operands)", output.what);
    }
    prog.into_case(format!("{name} [{rows}, {width}]"), rows, width)
}

struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Noise {
        Noise(seed | 1)
    }
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn below(&mut self, n: u32) -> u32 {
        ((self.unit() as f64 * f64::from(n)) as u32).min(n.saturating_sub(1))
    }
    fn word(&mut self) -> u32 {
        self.unit();
        (self.0 >> 17) as u32
    }
}

fn generate(noise: &mut Noise, dtype: Dtype, len: usize, source: Gen, row: u32) -> Value {
    let row = row as usize;
    match (dtype, source) {
        (Dtype::F32, Gen::Signed) => Value::F32(
            (0..len)
                .map(|_| match noise.below(16) {
                    0 => 0.0,
                    1 => -0.0,
                    // A coarse grid: ties within and across rows.
                    2..=5 => (noise.below(17) as f32 - 8.0) * 0.25,
                    _ => (noise.unit() - 0.5) * 8.0,
                })
                .collect(),
        ),
        (Dtype::F32, Gen::Positive) => {
            Value::F32((0..len).map(|_| 0.01 + noise.unit() * 8.0).collect())
        }
        (Dtype::F32, Gen::Probs) => {
            let mut out = Vec::with_capacity(len);
            for _ in 0..len.div_ceil(row) {
                let logits: Vec<f32> = (0..row).map(|_| (noise.unit() - 0.5) * 12.0).collect();
                let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = logits.iter().map(|&l| (l - peak).exp()).collect();
                let sum: f32 = e.iter().sum();
                out.extend(e.iter().map(|&x| x / sum));
            }
            out.truncate(len);
            Value::F32(out)
        }
        (Dtype::F32, Gen::Mass) => Value::F32(
            (0..len)
                .map(|_| [0.0, 0.3, 0.5, 0.9, 0.95, 1.0, 1.5][noise.below(7) as usize])
                .collect(),
        ),
        (Dtype::I32, Gen::Int(span)) => Value::I32(
            (0..len)
                .map(|_| noise.below(2 * span as u32) as i32 - span)
                .collect(),
        ),
        (Dtype::I32, Gen::Index(n)) => {
            Value::I32((0..len).map(|_| noise.below(n + 2) as i32 - 1).collect())
        }
        (Dtype::U32, Gen::Below(u32::MAX)) => Value::U32((0..len).map(|_| noise.word()).collect()),
        (Dtype::U32, Gen::Below(n)) => Value::U32((0..len).map(|_| noise.below(n)).collect()),
        (Dtype::Bool, Gen::Bool) => Value::Bool((0..len).map(|_| noise.below(2) == 1).collect()),
        other => panic!("no generator for {other:?}"),
    }
}

fn wire_value(bytes: &[u8], dtype: Dtype, len: usize) -> Value {
    let words = || bytes.as_chunks::<4>().0.iter().take(len);
    match dtype {
        Dtype::F32 => Value::F32(words().map(|c| f32::from_le_bytes(*c)).collect()),
        Dtype::I32 => Value::I32(words().map(|c| i32::from_le_bytes(*c)).collect()),
        Dtype::U32 => Value::U32(words().map(|c| u32::from_le_bytes(*c)).collect()),
        _ => Value::Bool((0..len).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect()),
    }
}

/// A value as a channel cell holds it: booleans a bit each.
fn wire_bytes(value: &Value) -> Vec<u8> {
    match value {
        Value::Bool(bits) => {
            let mut bytes = vec![0u8; bits.len().div_ceil(8)];
            for (i, &bit) in bits.iter().enumerate() {
                bytes[i / 8] |= u8::from(bit) << (i % 8);
            }
            bytes
        }
        other => other.to_le_bytes(),
    }
}

/// Distance in ulps over the ordered integer line of the floats.
fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let bits = x.to_bits() as i64;
        if bits < 0x8000_0000 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    key(a).abs_diff(key(b))
}

fn agree(got: &Value, want: &Value, check: Check) -> Result<(), String> {
    match (got, want, check) {
        (Value::F32(g), Value::F32(w), _) => {
            if g.len() != w.len() {
                return Err(format!("{} elements against {}", g.len(), w.len()));
            }
            for (i, (&x, &y)) in g.iter().zip(w).enumerate() {
                let fine = (x.is_nan() && y.is_nan())
                    || match check {
                        Check::Exact => x.to_bits() == y.to_bits(),
                        Check::Ulps(n) => ulps(x, y) <= u64::from(n),
                        Check::Scaled(n) => {
                            (x - y).abs() <= n as f32 * f32::EPSILON * y.abs().max(1.0)
                        }
                    };
                if !fine {
                    let wrong = g
                        .iter()
                        .zip(w)
                        .filter(|(x, y)| x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()))
                        .count();
                    return Err(format!(
                        "[{i}]: device {x:e} ({:#010x}) vs interpreter {y:e} ({:#010x}); {wrong} of {} differ",
                        x.to_bits(),
                        y.to_bits(),
                        g.len()
                    ));
                }
            }
            Ok(())
        }
        _ if got == want => Ok(()),
        (got, want, _) => {
            let (g, w) = (format!("{got:?}"), format!("{want:?}"));
            let at = g
                .bytes()
                .zip(w.bytes())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let lo = at.saturating_sub(60);
            Err(format!(
                "device …{}… vs interpreter …{}…",
                &g[lo..(at + 60).min(g.len())],
                &w[lo..(at + 60).min(w.len())]
            ))
        }
    }
}

fn bind(case: &Case) -> Result<BoundTrace, String> {
    crate::bound::logits(case.container.clone(), case.width)
        .map_err(|e| format!("the program does not bind: {e:?}"))
}

struct Lane {
    seeds: Vec<(u32, Value)>,
    logits: Option<Vec<f32>>,
}

fn lane_inputs(case: &Case, seed: u64, lane: u32) -> Lane {
    let mut noise = Noise::new(seed ^ (u64::from(lane) << 32) ^ u64::from(case.width));
    let seeds = case
        .inputs
        .iter()
        .map(|input| {
            let value = generate(&mut noise, input.dtype, input.len, input.source, input.row);
            (input.chan, value)
        })
        .collect();
    let logits = case.logits.then(|| {
        let Value::F32(plane) = generate(
            &mut noise,
            Dtype::F32,
            (case.rows * case.width) as usize,
            Gen::Signed,
            case.width,
        ) else {
            unreachable!()
        };
        plane
    });
    Lane { seeds, logits }
}

fn reference(bound: &BoundTrace, case: &Case, lane: &Lane) -> Vec<Value> {
    let mut instance = Instance::new(bound, &lane.seeds)
        .unwrap_or_else(|e| panic!("{}: the interpreter instantiates: {e:?}", case.name));
    let inputs = PassInputs {
        logits: lane.logits.clone().map(Value::F32),
        ..Default::default()
    };
    let stepped = instance
        .step(bound, &inputs, &mut NoKernels)
        .unwrap_or_else(|e| panic!("{}: the interpreter steps: {e:?}", case.name));
    assert!(
        stepped.committed,
        "{}: the reference did not commit",
        case.name
    );
    case.outputs
        .iter()
        .map(|o| {
            instance
                .host_take(bound, o.chan)
                .expect("a reference output")
        })
        .collect()
}

struct Rig {
    context: Context,
    plane: Plane,
}

impl Rig {
    fn register(&mut self, bound: &BoundTrace, hash: u64) -> Result<engine::ProgramId, String> {
        let stages = compile_bound(bound);
        let backend =
            Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
        let registration = ProgramRegistration {
            program_hash: hash,
            emitted_kernels: emit_program(backend, &stages, bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch: eta_compiler::codegen::launch::build(bound, &stages),
            ..Default::default()
        };
        self.plane
            .register(&self.context, &registration)
            .map_err(|e| format!("the program does not compile for the device: {e:?}"))
    }

    /// One wave of `lanes.len()` lanes of `program`, each lane's outputs.
    fn fire(
        &mut self,
        program: engine::ProgramId,
        bound: &BoundTrace,
        case: &Case,
        lanes: &[Lane],
    ) -> Result<Vec<Vec<Vec<u8>>>, String> {
        let count = bound.container.channels.len();
        let mut instances: Vec<InstanceId> = Vec::new();
        let mut planes: Vec<Buffer> = Vec::new();
        for (index, lane) in lanes.iter().enumerate() {
            let wire: Vec<(u32, Vec<u8>)> = lane
                .seeds
                .iter()
                .map(|(c, v)| (*c, wire_bytes(v)))
                .collect();
            let ids: Vec<ChannelId> = (1..=count as u64)
                .map(|c| ChannelId(c + 1000 * index as u64))
                .collect();
            let instance = self
                .plane
                .bind(
                    program,
                    &wire,
                    Extents {
                        row_count: case.rows,
                        sampled_rows: case.rows,
                        query_len: case.rows,
                        ..Extents::default()
                    },
                    GeometryClass::Host,
                    &vec![None; count],
                    &ids,
                    self.context.stream(),
                )
                .unwrap_or_else(|e| panic!("{}: the instance binds: {e:?}", case.name));
            if let Some(logits) = &lane.logits {
                let mut buffer = Buffer::zeroed(logits.len() * 4).expect("a logits plane");
                let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
                buffer.write(0, &bytes).expect("the plane uploads");
                self.plane
                    .bind_intrinsic(
                        instance,
                        IntrinsicId::Logits,
                        0,
                        buffer.ptr(),
                        INTRINSIC_STORAGE_F32,
                        case.width,
                        case.width,
                        0,
                    )
                    .expect("the logits bind");
                planes.push(buffer);
            }
            instances.push(instance);
        }
        for &instance in &instances {
            let launched = self
                .plane
                .stage(instance, Stage::Epilogue)
                .unwrap_or_else(|e| panic!("{}: the lane stages: {e:?}", case.name));
            assert!(
                matches!(launched, Launched::Airborne),
                "{}: a lane was refused",
                case.name
            );
        }
        self.plane
            .fly(&self.context)
            .unwrap_or_else(|e| panic!("{}: the wave flies: {e:?}", case.name));
        self.plane.land(&self.context).expect("the wave lands");
        self.context.synchronize().expect("the device synchronizes");
        let mut declined = None;
        let mut outputs = Vec::new();
        for &instance in &instances {
            let fired = self
                .plane
                .settle_launched(instance)
                .unwrap_or_else(|e| panic!("{}: the lane settles: {e:?}", case.name));
            if !matches!(fired, Fired::Committed) {
                declined = Some(format!("the device fire did not commit: {fired:?}"));
                continue;
            }
            let session = self.plane.instance_mut(instance).expect("the instance");
            outputs.push(
                case.outputs
                    .iter()
                    .map(|o| {
                        session
                            .take(o.chan)
                            .expect("a channel reads")
                            .expect("the channel holds a cell")
                    })
                    .collect(),
            );
        }
        for instance in instances {
            self.plane
                .close_instance(instance)
                .expect("the lane closes");
        }
        drop(planes);
        match declined {
            Some(why) => Err(why),
            None => Ok(outputs),
        }
    }
}

fn coverage(cases: &[Case]) {
    let mut seen = BTreeSet::new();
    for case in cases {
        for op in case.container.stages.iter().flat_map(|s| &s.ops) {
            seen.insert(op.tag());
        }
    }
    let missing: Vec<&str> = OP_TABLE
        .iter()
        .filter(|row| !EXEMPT.contains(&row.tag) && !seen.contains(&row.tag))
        .map(|row| row.name)
        .collect();
    assert!(missing.is_empty(), "ops no case runs: {missing:?}");
}

#[test]
fn every_op_agrees_with_the_host_interpreter() {
    let cases: Vec<Case> = OP_TABLE
        .iter()
        .filter(|row| !EXEMPT.contains(&row.tag))
        .flat_map(|row| SHAPES.iter().map(|&(r, w)| case(row.tag, r, w)))
        .collect();
    coverage(&cases);
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    // A substring of a case's name narrows the run, for a failure's repro.
    let only = std::env::var("PIE_EVERY_OP_ONLY").ok();
    let mut rig = Rig {
        context: Context::bind(0, None).expect("a CUDA context"),
        plane: Plane::new(Disk::disabled()),
    };
    let mut failures: Vec<String> = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        if only
            .as_ref()
            .is_some_and(|only| !case.name.contains(only.as_str()))
        {
            continue;
        }
        let bound = match bind(case) {
            Ok(bound) => bound,
            Err(why) => {
                failures.push(format!("{}: {why}", case.name));
                continue;
            }
        };
        let program = match rig.register(&bound, 0xe0_0000 + index as u64) {
            Ok(program) => program,
            Err(why) => {
                failures.push(format!("{}: {why}", case.name));
                continue;
            }
        };
        let mut failed = false;
        for &seed in SEEDS {
            for &lanes in LANES {
                if failed {
                    break;
                }
                let inputs: Vec<Lane> = (0..lanes).map(|l| lane_inputs(case, seed, l)).collect();
                let got = match rig.fire(program, &bound, case, &inputs) {
                    Ok(got) => got,
                    Err(why) => {
                        failures.push(format!(
                            "{} — seed {seed:#x}, {lanes} lane(s): {why}",
                            case.name
                        ));
                        failed = true;
                        continue;
                    }
                };
                for (lane, (input, got)) in inputs.iter().zip(&got).enumerate() {
                    let want = reference(&bound, case, input);
                    for ((output, bytes), want) in case.outputs.iter().zip(got).zip(&want) {
                        let got = wire_value(bytes, output.dtype, output.len);
                        if let Err(why) = agree(&got, want, output.check) {
                            failures.push(format!(
                                "{} {} — seed {seed:#x}, lane {lane} of {lanes}: {why}",
                                case.name, output.what
                            ));
                            failed = true;
                        }
                    }
                    if failed {
                        break;
                    }
                }
            }
        }
        rig.plane
            .close_program(program)
            .expect("the program closes");
    }
    assert!(
        failures.is_empty(),
        "{} disagreement(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
