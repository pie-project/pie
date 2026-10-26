use alloc::vec;
use alloc::vec::Vec;

use crate::op::{Op, tags};
use crate::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId, from_wire, wire_dtype};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpWire {
    pub tag: u8,
    pub chan: i64,
    pub name_idx: u16,
    pub args: Vec<ValueId>,
    pub results: u32,
    pub imm: u32,
    pub imm2: u32,
    pub imm3: u32,
    pub pred_tag: u8,
    pub pred_payload: u32,
    pub lit_dtype: u8,
    pub lit_bits: u32,
    pub dtype: u8,
    pub shape: Vec<u32>,
    pub kind: u8,
}

pub mod predicate_tags {
    pub const RANK_LE: u8 = 0;
    pub const CUMMASS_LE: u8 = 1;
    pub const PROB_GE: u8 = 2;
}

fn wire_shape(shape: &Shape) -> Vec<u32> {
    shape.dims().to_vec()
}

impl OpWire {
    pub fn of(op: &Op) -> Self {
        let mut wire = OpWire {
            tag: op.tag(),
            chan: -1,
            results: op.result_count(),
            ..OpWire::default()
        };
        match *op {
            Op::PivotThreshold { input, predicate } => {
                wire.args = vec![input];
                let (tag, payload) = match predicate {
                    Predicate::RankLe(value) => (predicate_tags::RANK_LE, value),
                    Predicate::CummassLe(value) => (predicate_tags::CUMMASS_LE, value),
                    Predicate::ProbGe(value) => (predicate_tags::PROB_GE, value),
                };
                wire.pred_tag = tag;
                wire.pred_payload = payload;
            }
            Op::TopK { input, k } => {
                wire.args = vec![input];
                wire.imm = k;
            }
            Op::Iota { len } => wire.imm = len,
            Op::Rng {
                stream,
                ref shape,
                kind,
            } => {
                wire.imm = stream;
                wire.shape = wire_shape(shape);
                wire.kind = kind as u8;
            }
            Op::RngKeyed {
                state,
                ref shape,
                kind,
            } => {
                wire.args = vec![state];
                wire.shape = wire_shape(shape);
                wire.kind = kind as u8;
            }
            Op::Const(literal) => {
                wire.lit_dtype = wire_dtype(literal.dtype());
                wire.lit_bits = match literal {
                    Literal::F32(value) => value.to_bits(),
                    Literal::I32(value) => value as u32,
                    Literal::U32(value) => value,
                    Literal::Bool(value) => u32::from(value),
                };
            }
            Op::Cast { value, dtype } => {
                wire.args = vec![value];
                wire.dtype = wire_dtype(dtype);
            }
            Op::Broadcast { value, ref shape } | Op::Reshape { value, ref shape } => {
                wire.args = vec![value];
                wire.shape = wire_shape(shape);
            }
            Op::CausalMask { positions, len } => {
                wire.args = vec![positions];
                wire.imm = len;
            }
            Op::SlidingWindowMask {
                positions,
                len,
                window,
            } => {
                wire.args = vec![positions];
                wire.imm = len;
                wire.imm2 = window;
            }
            Op::SinkWindowMask {
                positions,
                len,
                sink,
                window,
            } => {
                wire.args = vec![positions];
                wire.imm = len;
                wire.imm2 = sink;
                wire.imm3 = window;
            }
            Op::ChanTake(chan) | Op::ChanRead(chan) => wire.chan = i64::from(chan),
            Op::ChanPut { chan, value } => {
                wire.chan = i64::from(chan);
                wire.args = vec![value];
            }
            Op::Output {
                index,
                lane,
                ref shape,
                dtype,
            } => {
                wire.name_idx = index;
                wire.imm = lane;
                wire.shape = wire_shape(shape);
                wire.dtype = wire_dtype(dtype);
            }
            Op::KernelCall {
                name,
                ref args,
                ref shape,
                dtype,
            } => {
                wire.name_idx = name;
                wire.args = args.clone();
                wire.shape = wire_shape(shape);
                wire.dtype = wire_dtype(dtype);
            }
            Op::SinkCall { name, ref args } => {
                wire.name_idx = name;
                wire.args = args.clone();
            }
            _ => wire.args = op.operands(),
        }
        wire
    }

    pub fn of_all(ops: &[Op]) -> Vec<OpWire> {
        ops.iter().map(OpWire::of).collect()
    }

    pub fn channel(&self) -> Option<u32> {
        u32::try_from(self.chan).ok()
    }

    fn shape(&self) -> Option<Shape> {
        Shape::new(&self.shape)
    }

    fn dtype(&self) -> Option<Dtype> {
        from_wire(self.dtype)
    }

    fn rng_kind(&self) -> Option<RngKind> {
        match self.kind {
            0 => Some(RngKind::Uniform),
            1 => Some(RngKind::Gumbel),
            2 => Some(RngKind::Normal),
            _ => None,
        }
    }

    fn literal(&self) -> Option<Literal> {
        Some(match from_wire(self.lit_dtype)? {
            Dtype::F32 => Literal::F32(f32::from_bits(self.lit_bits)),
            Dtype::I32 => Literal::I32(self.lit_bits as i32),
            Dtype::U32 => Literal::U32(self.lit_bits),
            Dtype::Bool => Literal::Bool(self.lit_bits != 0),
            _ => return None,
        })
    }

    fn predicate(&self) -> Option<Predicate> {
        Some(match self.pred_tag {
            predicate_tags::RANK_LE => Predicate::RankLe(self.pred_payload),
            predicate_tags::CUMMASS_LE => Predicate::CummassLe(self.pred_payload),
            predicate_tags::PROB_GE => Predicate::ProbGe(self.pred_payload),
            _ => return None,
        })
    }

    fn arg(&self, index: usize) -> Option<ValueId> {
        self.args.get(index).copied()
    }

    pub fn to_op(&self) -> Option<Op> {
        let a0 = || self.arg(0);
        let a1 = || self.arg(1);
        let a2 = || self.arg(2);
        Some(match self.tag {
            tags::EXP => Op::Exp(a0()?),
            tags::LOG => Op::Log(a0()?),
            tags::NEG => Op::Neg(a0()?),
            tags::RECIP => Op::Recip(a0()?),
            tags::SIN => Op::Sin(a0()?),
            tags::COS => Op::Cos(a0()?),
            tags::SQRT => Op::Sqrt(a0()?),
            tags::RSQRT => Op::Rsqrt(a0()?),
            tags::ABS => Op::Abs(a0()?),
            tags::SIGN => Op::Sign(a0()?),
            tags::CAST => Op::Cast {
                value: a0()?,
                dtype: self.dtype()?,
            },
            tags::ADD => Op::Add(a0()?, a1()?),
            tags::SUB => Op::Sub(a0()?, a1()?),
            tags::MUL => Op::Mul(a0()?, a1()?),
            tags::DIV => Op::Div(a0()?, a1()?),
            tags::MAX_ELEM => Op::MaxElem(a0()?, a1()?),
            tags::MIN_ELEM => Op::MinElem(a0()?, a1()?),
            tags::GT => Op::Gt(a0()?, a1()?),
            tags::GE => Op::Ge(a0()?, a1()?),
            tags::EQ => Op::Eq(a0()?, a1()?),
            tags::NE => Op::Ne(a0()?, a1()?),
            tags::LT => Op::Lt(a0()?, a1()?),
            tags::LE => Op::Le(a0()?, a1()?),
            tags::AND => Op::And(a0()?, a1()?),
            tags::OR => Op::Or(a0()?, a1()?),
            tags::NOT => Op::Not(a0()?),
            tags::REM => Op::Rem(a0()?, a1()?),
            tags::SELECT => Op::Select {
                cond: a0()?,
                a: a1()?,
                b: a2()?,
            },
            tags::REDUCE_SUM => Op::ReduceSum(a0()?),
            tags::REDUCE_MAX => Op::ReduceMax(a0()?),
            tags::REDUCE_MIN => Op::ReduceMin(a0()?),
            tags::REDUCE_ARGMAX => Op::ReduceArgmax(a0()?),
            tags::BROADCAST => Op::Broadcast {
                value: a0()?,
                shape: self.shape()?,
            },
            tags::RESHAPE => Op::Reshape {
                value: a0()?,
                shape: self.shape()?,
            },
            tags::TRANSPOSE => Op::Transpose(a0()?),
            tags::CUMSUM => Op::CumSum(a0()?),
            tags::CUMPROD => Op::CumProd(a0()?),
            tags::SORT_DESC => Op::SortDesc(a0()?),
            tags::TOP_K => Op::TopK {
                input: a0()?,
                k: self.imm,
            },
            tags::MATMUL => Op::MatMul(a0()?, a1()?),
            tags::PIVOT_THRESHOLD => Op::PivotThreshold {
                input: a0()?,
                predicate: self.predicate()?,
            },
            tags::GATHER => Op::Gather {
                src: a0()?,
                idx: a1()?,
            },
            tags::GATHER_ROW => Op::GatherRow {
                src: a0()?,
                idx: a1()?,
            },
            tags::SCATTER_ADD => Op::ScatterAdd {
                base: a0()?,
                idx: a1()?,
                vals: a2()?,
            },
            tags::SCATTER_SET => Op::ScatterSet {
                base: a0()?,
                idx: a1()?,
                vals: a2()?,
            },
            tags::IOTA => Op::Iota { len: self.imm },
            tags::MASK_APPLY_PACKED => Op::MaskApply {
                logits: a0()?,
                mask: a1()?,
            },
            tags::CAUSAL_MASK => Op::CausalMask {
                positions: a0()?,
                len: self.imm,
            },
            tags::SLIDING_WINDOW_MASK => Op::SlidingWindowMask {
                positions: a0()?,
                len: self.imm,
                window: self.imm2,
            },
            tags::SINK_WINDOW_MASK => Op::SinkWindowMask {
                positions: a0()?,
                len: self.imm,
                sink: self.imm2,
                window: self.imm3,
            },
            tags::RNG => Op::Rng {
                stream: self.imm,
                shape: self.shape()?,
                kind: self.rng_kind()?,
            },
            tags::RNG_KEYED => Op::RngKeyed {
                state: a0()?,
                shape: self.shape()?,
                kind: self.rng_kind()?,
            },
            tags::CONST => Op::Const(self.literal()?),
            tags::CHAN_TAKE => Op::ChanTake(self.channel()?),
            tags::CHAN_READ => Op::ChanRead(self.channel()?),
            tags::CHAN_PUT => Op::ChanPut {
                chan: self.channel()?,
                value: a0()?,
            },
            tags::OUTPUT => Op::Output {
                index: self.name_idx,
                lane: self.imm,
                shape: self.shape()?,
                dtype: self.dtype()?,
            },
            tags::KERNEL_CALL => Op::KernelCall {
                name: self.name_idx,
                args: self.args.clone(),
                shape: self.shape()?,
                dtype: self.dtype()?,
            },
            tags::SINK_CALL => Op::SinkCall {
                name: self.name_idx,
                args: self.args.clone(),
            },
            _ => return None,
        })
    }
}

pub fn result_bases(ops: &[OpWire]) -> Vec<u32> {
    let mut bases = Vec::with_capacity(ops.len());
    let mut next_value = 0u32;
    for op in ops {
        bases.push(next_value);
        next_value = next_value.wrapping_add(op.results);
    }
    bases
}
