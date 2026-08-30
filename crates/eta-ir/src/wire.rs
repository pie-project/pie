//! [`OpWire`] — the flat wire record of one [`Op`], and the two projections
//! between it and the typed enum.
//!
//! An op exists in three forms: the typed [`Op`] enum passes write, the bytes
//! the container carries, and the flat record a backend emitter switches on.
//! One layout declaration ([`OpSpec::wire`](crate::op::OpSpec::wire)) and one
//! pair of projections connect all three:
//!
//! ```text
//!   Op  ──OpWire::of──>  OpWire  ──container::encode_op──>  bytes
//!   Op  <──from_wire──   OpWire  <──container::decode_op──  bytes
//! ```
//!
//! [`OpWire::of`] and [`OpWire::to_op`] are exact inverses, checked over every
//! row of the op table by `the_wire_record_roundtrips_every_op`.
//!
//! Adding a payload field means adding a [`WireField`](crate::op::WireField)
//! to the op's row — never a case to a projection here. A field added to a
//! projection alone is invisible: it encodes and decodes consistently within
//! this crate while every other reader of the byte stream disagrees about
//! where the following fields begin.

use alloc::vec;
use alloc::vec::Vec;

use crate::op::{IntrinsicId, Op, tags};
use crate::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId, from_wire, wire_dtype};

/// The wire view of one op: its tag plus the payload fields
/// [`OpSpec::wire`](crate::op::OpSpec::wire) declares, flattened.
///
/// Positional immediates land in `imm`, `imm2`, `imm3` in declaration order,
/// so `sliding_window_mask`'s `len`/`window` are `imm`/`imm2` and
/// `sink_window_mask`'s `len`/`sink`/`window` are `imm`/`imm2`/`imm3`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpWire {
    /// The op's wire tag; what selects the layout the rest of this record
    /// was filled from.
    pub tag: u8,
    /// `chan_take` / `chan_read` / `chan_put` target, else `-1`.
    pub chan: i64,
    /// `kernel_call` / `sink_call` name-table index.
    pub name_idx: u16,
    /// Value-id operands **as the container encodes them**: this is not
    /// [`Op::operands`]. `pivot_threshold` encodes its predicate payload in a
    /// separate field, so `args` holds the input alone.
    pub args: Vec<ValueId>,
    /// SSA ids defined.
    pub results: u32,
    /// `intrinsic_val` id.
    pub intr: u16,
    /// First trace-known immediate: `top_k` k / `iota` len / `rng` stream /
    /// mask len.
    pub imm: u32,
    /// Second immediate: `sliding_window_mask` window / `sink_window_mask` sink.
    pub imm2: u32,
    /// Third immediate: `sink_window_mask` window.
    pub imm3: u32,
    /// `pivot_threshold` predicate tag.
    pub pred_tag: u8,
    /// `pivot_threshold` predicate payload (a value id for every tag).
    pub pred_payload: u32,
    /// `const` literal dtype.
    pub lit_dtype: u8,
    /// `const` literal raw bits, interpreted per `lit_dtype`.
    pub lit_bits: u32,
    /// `cast` / `rng` / `intrinsic_val` / `kernel_call` element dtype.
    pub dtype: u8,
    /// `broadcast` / `reshape` / `rng` / `intrinsic_val` / `kernel_call`
    /// target shape.
    pub shape: Vec<u32>,
    /// `rng` kind — 0 uniform, 1 gumbel.
    pub kind: u8,
}

/// The predicate wire tags, in [`Predicate`] declaration order.
pub mod predicate_tags {
    /// Keep the entries whose rank is at most `k`.
    pub const RANK_LE: u8 = 0;
    /// Keep the shortest prefix whose cumulative mass reaches `p`.
    pub const CUMMASS_LE: u8 = 1;
    /// Keep the entries whose probability is at least the threshold.
    pub const PROB_GE: u8 = 2;
}

fn wire_shape(shape: &Shape) -> Vec<u32> {
    shape.dims().to_vec()
}

impl OpWire {
    /// Project one op onto its wire fields.
    ///
    /// Every payload an op carries beyond its value operands must be set by a
    /// named arm here — the general arm at the bottom only fills `args`, so an
    /// op that reaches it having an immediate, a shape or a dtype would encode
    /// those as zeros. `the_wire_record_roundtrips_every_op` is what makes
    /// that a test failure rather than a silently wrong kernel.
    pub fn of(op: &Op) -> Self {
        let mut wire = OpWire {
            tag: op.tag(),
            chan: -1,
            results: op.result_count(),
            ..OpWire::default()
        };
        match *op {
            // `pivot_threshold` encodes `input` then a 5-byte predicate, so
            // the predicate operand is NOT an arg — unlike `Op::operands`.
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
            Op::IntrinsicVal {
                intr,
                ref shape,
                dtype,
            } => {
                wire.intr = intr as u16;
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

    /// Project a whole stage body.
    pub fn of_all(ops: &[Op]) -> Vec<OpWire> {
        ops.iter().map(OpWire::of).collect()
    }

    /// The channel index this op names, if any.
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
            _ => None,
        }
    }

    fn literal(&self) -> Option<Literal> {
        Some(match from_wire(self.lit_dtype)? {
            Dtype::F32 => Literal::F32(f32::from_bits(self.lit_bits)),
            Dtype::I32 => Literal::I32(self.lit_bits as i32),
            Dtype::U32 => Literal::U32(self.lit_bits),
            Dtype::Bool => Literal::Bool(self.lit_bits != 0),
            // Unreachable behind the gate: `from_wire` answers only over
            // `types::WIRE_ORDER`, and a `Literal` exists for each of those
            // four. It is a `None` and not an `unreachable!` because this
            // function's whole job is to answer `None` for a byte that does
            // not name a literal.
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

    /// Rebuild the typed op — the inverse of [`OpWire::of`].
    ///
    /// `None` when a field names something the vocabulary does not have (an
    /// unknown dtype, intrinsic, rng kind or predicate tag) or when an operand
    /// the tag requires is missing. The container's decoder validates those
    /// bytes as it reads them so it can name the offending field, which leaves
    /// this a total function on records it produced.
    pub fn to_op(&self) -> Option<Op> {
        let a0 = || self.arg(0);
        let a1 = || self.arg(1);
        let a2 = || self.arg(2);
        Some(match self.tag {
            tags::EXP => Op::Exp(a0()?),
            tags::LOG => Op::Log(a0()?),
            tags::NEG => Op::Neg(a0()?),
            tags::RECIP => Op::Recip(a0()?),
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
            tags::INTRINSIC_VAL => Op::IntrinsicVal {
                intr: IntrinsicId::from_u16(self.intr)?,
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

/// `bases[node]` — the SSA id each op's first result defines.
pub fn result_bases(ops: &[OpWire]) -> Vec<u32> {
    let mut bases = Vec::with_capacity(ops.len());
    let mut next_value = 0u32;
    for op in ops {
        bases.push(next_value);
        next_value = next_value.wrapping_add(op.results);
    }
    bases
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{OP_TABLE, WireField, representatives};

    /// [`OpWire::of`] and [`OpWire::to_op`] must be inverses over the whole op
    /// table.
    ///
    /// This is what lets the container encode and decode through one layout
    /// declaration instead of two hand-written walks. A field `of` forgets to
    /// set comes back as a default here, and a field `to_op` reads from the
    /// wrong slot comes back permuted — either way the op that returns is not
    /// the op that went in.
    #[test]
    fn the_wire_record_roundtrips_every_op() {
        let ops = representatives();
        assert_eq!(ops.len(), OP_TABLE.len());
        for op in ops {
            let wire = OpWire::of(&op);
            assert_eq!(
                wire.to_op().as_ref(),
                Some(&op),
                "{op:?} did not survive the wire projection"
            );
        }
    }

    /// The layout column must describe the record the projection actually
    /// fills: as many `Value` fields as `args` (before any variadic `Args`),
    /// as many `Imm` fields as populated immediate slots, and a `Shape` /
    /// `Dtype` / `Chan` field exactly when the record carries one.
    ///
    /// Without this the two could disagree in the direction a roundtrip
    /// cannot see — `of` and `to_op` agreeing with each other while the bytes
    /// in between carry a field neither of them names.
    #[test]
    fn the_layout_column_describes_the_record() {
        for op in representatives() {
            let wire = OpWire::of(&op);
            let layout = crate::op::spec(wire.tag).expect("every op has a row").wire;
            let count = |field: WireField| layout.iter().filter(|&&f| f == field).count();

            if !layout.contains(&WireField::Args) {
                assert_eq!(
                    count(WireField::Value),
                    wire.args.len(),
                    "{op:?}: layout declares {} value fields, record has {} args",
                    count(WireField::Value),
                    wire.args.len()
                );
            }
            assert_eq!(
                layout.contains(&WireField::Chan),
                wire.chan >= 0,
                "{op:?}: Chan field and record disagree"
            );
            assert_eq!(
                layout.contains(&WireField::Shape),
                !wire.shape.is_empty(),
                "{op:?}: Shape field and record disagree"
            );
            let imms = count(WireField::Imm);
            assert!(
                imms <= 3,
                "{op:?}: {imms} immediates, but the record has three slots"
            );
            if imms < 3 {
                assert_eq!(wire.imm3, 0, "{op:?}: imm3 set but not declared");
            }
            if imms < 2 {
                assert_eq!(wire.imm2, 0, "{op:?}: imm2 set but not declared");
            }
        }
    }

    /// A variadic `Args` field consumes the rest of the body, so nothing may
    /// follow it.
    #[test]
    fn variadic_args_come_last() {
        for row in OP_TABLE {
            if let Some(position) = row.wire.iter().position(|&f| f == WireField::Args) {
                assert_eq!(
                    position,
                    row.wire.len() - 1,
                    "`{}` has fields after its variadic Args",
                    row.name
                );
            }
        }
    }
}
