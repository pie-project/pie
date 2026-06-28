//! The flat versioned bytecode: the internal host↔driver encoding.
//!
//! v4 = the **shape-typed, binding-free** layout: typed input slots + a flat SSA
//! op list (`input`/`const` as op tags) + bare output ids (with kind kept for
//! host marshaling), shapes as `rank:u8 | dims[rank]:u32`. The structure differs
//! from v1–v3, so a v4 reader accepts only `version == 4`. Everything is
//! little-endian; binding (logits/tensor) is attach-time, **not** in the
//! bytecode.

use alloc::vec::Vec;
use core::fmt;

use crate::types::*;
use crate::validate::ValidationError;
use crate::{MAGIC, VERSION};

// -- dtype tags --
const DT_F32: u8 = 0;
const DT_I32: u8 = 1;
const DT_U32: u8 = 2;
const DT_BOOL: u8 = 3;

/// `InputDecl` readiness rides the high bits of the dtype byte (DType tags are
/// `0..=3`, so bits 2..=7 are free). Clear ⇒ `Submit` — so v4 bytecode (which
/// never set these) decodes additively as `Submit`. Bit 7 ⇒ `Late`; bit 6 ⇒
/// `SelfSpecDraftInput` (#31 self-spec verify draft — a distinct device-resident
/// late role). Existing `Submit`/`Late` encodings are unchanged (additive).
const READY_LATE_BIT: u8 = 0x80;
/// Bit 6 of the dtype byte: `Readiness::SelfSpecDraftInput` (#31). Distinct from
/// `READY_LATE_BIT` so the role round-trips losslessly; checked first on decode.
const READY_SELFSPEC_BIT: u8 = 0x40;

// -- predicate tags --
const PR_RANKLE: u8 = 0;
const PR_CUMMASSLE: u8 = 1;
const PR_PROBGE: u8 = 2;

// -- rng kind tags --
const RK_UNIFORM: u8 = 0;
const RK_GUMBEL: u8 = 1;

// -- op tags (see BYTECODE.md) --
const OP_INPUT: u8 = 0x80;
const OP_CONST: u8 = 0x81;

const OP_EXP: u8 = 0x01;
const OP_LOG: u8 = 0x02;
const OP_NEG: u8 = 0x03;
const OP_RECIP: u8 = 0x04;
const OP_ABS: u8 = 0x05;
const OP_SIGN: u8 = 0x06;

const OP_ADD: u8 = 0x10;
const OP_SUB: u8 = 0x11;
const OP_MUL: u8 = 0x12;
const OP_DIV: u8 = 0x13;
const OP_MAXELEM: u8 = 0x14;
const OP_MINELEM: u8 = 0x15;
const OP_GT: u8 = 0x16;
const OP_GE: u8 = 0x17;
const OP_EQ: u8 = 0x18;
const OP_SELECT: u8 = 0x20;

const OP_REDUCESUM: u8 = 0x30;
const OP_REDUCEMAX: u8 = 0x31;
const OP_REDUCEMIN: u8 = 0x32;
const OP_REDUCEARGMAX: u8 = 0x33;
const OP_BROADCAST: u8 = 0x38;

const OP_CUMSUM: u8 = 0x40;
const OP_CUMPROD: u8 = 0x41;

const OP_SORTDESC: u8 = 0x50;
const OP_PIVOTTHRESHOLD: u8 = 0x58;

const OP_GATHER: u8 = 0x60;
const OP_GATHERROW: u8 = 0x61;
const OP_SCATTERADD: u8 = 0x62;
const OP_SCATTERSET: u8 = 0x63;
const OP_MASKAPPLY: u8 = 0x65;

const OP_RNG: u8 = 0x70;

// ===========================================================================
// Encode
// ===========================================================================

/// Lower a [`SamplingProgram`] to bytecode. Does not validate.
pub fn encode(p: &SamplingProgram) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(&MAGIC);
    put_u16(&mut w, VERSION);
    put_u16(&mut w, 0); // flags
    put_u32(&mut w, p.inputs.len() as u32);
    put_u32(&mut w, p.ops.len() as u32);
    put_u32(&mut w, p.outputs.len() as u32);
    for inp in &p.inputs {
        // Readiness rides the high tag bits (additive): `SelfSpecDraftInput` (#31
        // self-spec verify draft, device-resident) gets its own bit so it round-trips
        // losslessly; `Late` keeps bit 7. The structured manifest is the source of
        // truth for the role; the bytecode bit mirrors it.
        let ready_bit = match inp.ready {
            Readiness::SelfSpecDraftInput => READY_SELFSPEC_BIT,
            Readiness::Late => READY_LATE_BIT,
            Readiness::Submit => 0,
        };
        w.push(dtype_tag(inp.dtype) | ready_bit);
        encode_shape(&mut w, inp.shape);
    }
    for op in &p.ops {
        encode_op(&mut w, op);
    }
    for o in &p.outputs {
        put_u32(&mut w, o.value);
        w.push(o.kind.to_u8());
    }
    w
}

fn encode_op(w: &mut Vec<u8>, op: &Op) {
    match *op {
        Op::Input(index) => {
            w.push(OP_INPUT);
            put_u32(w, index);
        }
        Op::Const(lit) => {
            w.push(OP_CONST);
            encode_literal(w, lit);
        }

        Op::Exp(a) => un(w, OP_EXP, a),
        Op::Log(a) => un(w, OP_LOG, a),
        Op::Neg(a) => un(w, OP_NEG, a),
        Op::Recip(a) => un(w, OP_RECIP, a),
        Op::Abs(a) => un(w, OP_ABS, a),
        Op::Sign(a) => un(w, OP_SIGN, a),

        Op::Add(a, b) => bin(w, OP_ADD, a, b),
        Op::Sub(a, b) => bin(w, OP_SUB, a, b),
        Op::Mul(a, b) => bin(w, OP_MUL, a, b),
        Op::Div(a, b) => bin(w, OP_DIV, a, b),
        Op::MaxElem(a, b) => bin(w, OP_MAXELEM, a, b),
        Op::MinElem(a, b) => bin(w, OP_MINELEM, a, b),
        Op::Gt(a, b) => bin(w, OP_GT, a, b),
        Op::Ge(a, b) => bin(w, OP_GE, a, b),
        Op::Eq(a, b) => bin(w, OP_EQ, a, b),
        Op::Select { cond, a, b } => {
            w.push(OP_SELECT);
            put_u32(w, cond);
            put_u32(w, a);
            put_u32(w, b);
        }

        Op::ReduceSum(v) => un(w, OP_REDUCESUM, v),
        Op::ReduceMax(v) => un(w, OP_REDUCEMAX, v),
        Op::ReduceMin(v) => un(w, OP_REDUCEMIN, v),
        Op::ReduceArgmax(v) => un(w, OP_REDUCEARGMAX, v),
        Op::Broadcast { value, shape } => {
            w.push(OP_BROADCAST);
            put_u32(w, value);
            encode_shape(w, shape);
        }

        Op::CumSum(v) => un(w, OP_CUMSUM, v),
        Op::CumProd(v) => un(w, OP_CUMPROD, v),

        Op::SortDesc(v) => un(w, OP_SORTDESC, v),
        Op::PivotThreshold { input, predicate } => {
            w.push(OP_PIVOTTHRESHOLD);
            put_u32(w, input);
            encode_predicate(w, predicate);
        }

        Op::Gather { src, idx } => bin(w, OP_GATHER, src, idx),
        Op::GatherRow { src, idx } => bin(w, OP_GATHERROW, src, idx),
        Op::MaskApply { logits, mask } => bin(w, OP_MASKAPPLY, logits, mask),
        Op::ScatterAdd { base, idx, vals } => {
            w.push(OP_SCATTERADD);
            put_u32(w, base);
            put_u32(w, idx);
            put_u32(w, vals);
        }
        Op::ScatterSet { base, idx, vals } => {
            w.push(OP_SCATTERSET);
            put_u32(w, base);
            put_u32(w, idx);
            put_u32(w, vals);
        }

        Op::Rng { stream, shape, kind } => {
            w.push(OP_RNG);
            put_u32(w, stream);
            encode_shape(w, shape);
            w.push(match kind {
                RngKind::Uniform => RK_UNIFORM,
                RngKind::Gumbel => RK_GUMBEL,
            });
        }
    }
}

fn encode_predicate(w: &mut Vec<u8>, pred: Predicate) {
    match pred {
        Predicate::RankLe(k) => {
            w.push(PR_RANKLE);
            put_u32(w, k);
        }
        Predicate::CummassLe(v) => {
            w.push(PR_CUMMASSLE);
            put_u32(w, v);
        }
        Predicate::ProbGe(v) => {
            w.push(PR_PROBGE);
            put_u32(w, v);
        }
    }
}

/// Shape = `rank:u8 | dims[rank]:u32`.
fn encode_shape(w: &mut Vec<u8>, shape: Shape) {
    w.push(shape.rank() as u8);
    for &d in shape.dims() {
        put_u32(w, d);
    }
}

/// Literal = `dtype:u8 | value:u32`.
fn encode_literal(w: &mut Vec<u8>, lit: Literal) {
    match lit {
        Literal::F32(x) => {
            w.push(DT_F32);
            put_u32(w, x.to_bits());
        }
        Literal::I32(x) => {
            w.push(DT_I32);
            put_u32(w, x as u32);
        }
        Literal::U32(x) => {
            w.push(DT_U32);
            put_u32(w, x);
        }
        Literal::Bool(b) => {
            w.push(DT_BOOL);
            put_u32(w, b as u32);
        }
    }
}

fn un(w: &mut Vec<u8>, tag: u8, a: ValueId) {
    w.push(tag);
    put_u32(w, a);
}

fn bin(w: &mut Vec<u8>, tag: u8, a: ValueId, b: ValueId) {
    w.push(tag);
    put_u32(w, a);
    put_u32(w, b);
}

fn dtype_tag(d: DType) -> u8 {
    match d {
        DType::F32 => DT_F32,
        DType::I32 => DT_I32,
        DType::U32 => DT_U32,
        DType::Bool => DT_BOOL,
    }
}

fn put_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

// ===========================================================================
// Decode
// ===========================================================================

/// A bytecode decode failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    BadMagic,
    UnsupportedVersion(u16),
    UnexpectedEof,
    UnknownOpcode(u8),
    UnknownTag { what: &'static str, tag: u8 },
    RankTooLarge(u8),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::BadMagic => f.write_str("bad magic (expected \"PSIR\")"),
            DecodeError::UnsupportedVersion(v) => write!(f, "unsupported bytecode version {v}"),
            DecodeError::UnexpectedEof => f.write_str("unexpected end of buffer"),
            DecodeError::UnknownOpcode(t) => write!(f, "unknown opcode 0x{t:02x}"),
            DecodeError::UnknownTag { what, tag } => write!(f, "unknown {what} tag 0x{tag:02x}"),
            DecodeError::RankTooLarge(r) => write!(f, "shape rank {r} exceeds MAX_RANK"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn u8(&mut self) -> Result<u8, DecodeError> {
        let b = *self.buf.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }
    fn u16(&mut self) -> Result<u16, DecodeError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::UnexpectedEof)?;
        let s = self.buf.get(self.pos..end).ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(s)
    }
}

/// Parse bytecode back into the typed IR. Does not validate.
pub fn decode(bytes: &[u8]) -> Result<SamplingProgram, DecodeError> {
    let mut r = Reader::new(bytes);
    let magic = r.take(4)?;
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let version = r.u16()?;
    if version != VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    let _flags = r.u16()?;
    let n_inputs = r.u32()?;
    let n_ops = r.u32()?;
    let n_outputs = r.u32()?;

    let mut inputs = Vec::with_capacity(n_inputs as usize);
    for _ in 0..n_inputs {
        let raw = r.u8()?;
        // Check the self-spec bit first (distinct role), then the late bit.
        let ready = if raw & READY_SELFSPEC_BIT != 0 {
            Readiness::SelfSpecDraftInput
        } else if raw & READY_LATE_BIT != 0 {
            Readiness::Late
        } else {
            Readiness::Submit
        };
        let dtype = decode_dtype(raw & !(READY_LATE_BIT | READY_SELFSPEC_BIT))?;
        let shape = decode_shape(&mut r)?;
        inputs.push(InputDecl::with_ready(shape, dtype, ready));
    }
    let mut ops = Vec::with_capacity(n_ops as usize);
    for _ in 0..n_ops {
        ops.push(decode_op(&mut r)?);
    }
    let mut outputs = Vec::with_capacity(n_outputs as usize);
    for _ in 0..n_outputs {
        let value = r.u32()?;
        let kind_tag = r.u8()?;
        let kind = OutputKind::from_u8(kind_tag)
            .ok_or(DecodeError::UnknownTag { what: "output kind", tag: kind_tag })?;
        outputs.push(OutputDecl { value, kind });
    }
    Ok(SamplingProgram { inputs, ops, outputs })
}

/// Decode then validate in one call.
pub fn decode_validated(bytes: &[u8]) -> Result<SamplingProgram, DecodeErrorOrInvalid> {
    let p = decode(bytes).map_err(DecodeErrorOrInvalid::Decode)?;
    p.validate().map_err(DecodeErrorOrInvalid::Invalid)?;
    Ok(p)
}

/// Error union for [`decode_validated`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeErrorOrInvalid {
    Decode(DecodeError),
    Invalid(ValidationError),
}

impl fmt::Display for DecodeErrorOrInvalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeErrorOrInvalid::Decode(e) => write!(f, "decode: {e}"),
            DecodeErrorOrInvalid::Invalid(e) => write!(f, "invalid: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeErrorOrInvalid {}

fn decode_op(r: &mut Reader<'_>) -> Result<Op, DecodeError> {
    let tag = r.u8()?;
    let op = match tag {
        OP_INPUT => Op::Input(r.u32()?),
        OP_CONST => Op::Const(decode_literal(r)?),

        OP_EXP => Op::Exp(r.u32()?),
        OP_LOG => Op::Log(r.u32()?),
        OP_NEG => Op::Neg(r.u32()?),
        OP_RECIP => Op::Recip(r.u32()?),
        OP_ABS => Op::Abs(r.u32()?),
        OP_SIGN => Op::Sign(r.u32()?),

        OP_ADD => bin_op(r, Op::Add)?,
        OP_SUB => bin_op(r, Op::Sub)?,
        OP_MUL => bin_op(r, Op::Mul)?,
        OP_DIV => bin_op(r, Op::Div)?,
        OP_MAXELEM => bin_op(r, Op::MaxElem)?,
        OP_MINELEM => bin_op(r, Op::MinElem)?,
        OP_GT => bin_op(r, Op::Gt)?,
        OP_GE => bin_op(r, Op::Ge)?,
        OP_EQ => bin_op(r, Op::Eq)?,
        OP_SELECT => {
            let cond = r.u32()?;
            let a = r.u32()?;
            let b = r.u32()?;
            Op::Select { cond, a, b }
        }

        OP_REDUCESUM => Op::ReduceSum(r.u32()?),
        OP_REDUCEMAX => Op::ReduceMax(r.u32()?),
        OP_REDUCEMIN => Op::ReduceMin(r.u32()?),
        OP_REDUCEARGMAX => Op::ReduceArgmax(r.u32()?),
        OP_BROADCAST => {
            let value = r.u32()?;
            let shape = decode_shape(r)?;
            Op::Broadcast { value, shape }
        }

        OP_CUMSUM => Op::CumSum(r.u32()?),
        OP_CUMPROD => Op::CumProd(r.u32()?),

        OP_SORTDESC => Op::SortDesc(r.u32()?),
        OP_PIVOTTHRESHOLD => {
            let input = r.u32()?;
            let predicate = decode_predicate(r)?;
            Op::PivotThreshold { input, predicate }
        }

        OP_GATHER => {
            let src = r.u32()?;
            let idx = r.u32()?;
            Op::Gather { src, idx }
        }
        OP_GATHERROW => {
            let src = r.u32()?;
            let idx = r.u32()?;
            Op::GatherRow { src, idx }
        }
        OP_MASKAPPLY => {
            let logits = r.u32()?;
            let mask = r.u32()?;
            Op::MaskApply { logits, mask }
        }
        OP_SCATTERADD => {
            let base = r.u32()?;
            let idx = r.u32()?;
            let vals = r.u32()?;
            Op::ScatterAdd { base, idx, vals }
        }
        OP_SCATTERSET => {
            let base = r.u32()?;
            let idx = r.u32()?;
            let vals = r.u32()?;
            Op::ScatterSet { base, idx, vals }
        }

        OP_RNG => {
            let stream = r.u32()?;
            let shape = decode_shape(r)?;
            let kind = match r.u8()? {
                RK_UNIFORM => RngKind::Uniform,
                RK_GUMBEL => RngKind::Gumbel,
                t => return Err(DecodeError::UnknownTag { what: "rng kind", tag: t }),
            };
            Op::Rng { stream, shape, kind }
        }

        t => return Err(DecodeError::UnknownOpcode(t)),
    };
    Ok(op)
}

fn bin_op(r: &mut Reader<'_>, make: fn(ValueId, ValueId) -> Op) -> Result<Op, DecodeError> {
    let a = r.u32()?;
    let b = r.u32()?;
    Ok(make(a, b))
}

fn decode_predicate(r: &mut Reader<'_>) -> Result<Predicate, DecodeError> {
    Ok(match r.u8()? {
        PR_RANKLE => Predicate::RankLe(r.u32()?),
        PR_CUMMASSLE => Predicate::CummassLe(r.u32()?),
        PR_PROBGE => Predicate::ProbGe(r.u32()?),
        t => return Err(DecodeError::UnknownTag { what: "predicate", tag: t }),
    })
}

fn decode_shape(r: &mut Reader<'_>) -> Result<Shape, DecodeError> {
    let rank = r.u8()?;
    if rank as usize > MAX_RANK {
        return Err(DecodeError::RankTooLarge(rank));
    }
    let mut dims = [0u32; MAX_RANK];
    for d in dims.iter_mut().take(rank as usize) {
        *d = r.u32()?;
    }
    Ok(Shape::new(&dims[..rank as usize]).expect("rank checked"))
}

fn decode_literal(r: &mut Reader<'_>) -> Result<Literal, DecodeError> {
    let tag = r.u8()?;
    let bits = r.u32()?;
    Ok(match tag {
        DT_F32 => Literal::F32(f32::from_bits(bits)),
        DT_I32 => Literal::I32(bits as i32),
        DT_U32 => Literal::U32(bits),
        DT_BOOL => Literal::Bool(bits != 0),
        t => return Err(DecodeError::UnknownTag { what: "literal dtype", tag: t }),
    })
}

fn decode_dtype(tag: u8) -> Result<DType, DecodeError> {
    Ok(match tag {
        DT_F32 => DType::F32,
        DT_I32 => DType::I32,
        DT_U32 => DType::U32,
        DT_BOOL => DType::Bool,
        t => return Err(DecodeError::UnknownTag { what: "dtype", tag: t }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample_program() -> SamplingProgram {
        SamplingProgram {
            inputs: vec![
                InputDecl::new(Shape::vector(32_000), DType::F32),
                InputDecl::new(Shape::SCALAR, DType::U32),
            ],
            ops: vec![
                Op::Input(0),                 // 0 logits
                Op::Const(Literal::F32(0.7)), // 1 temp
                Op::Div(0, 1),                // 2
                Op::Exp(2),                   // 3
                Op::ReduceSum(3),             // 4
                Op::Div(3, 4),                // 5
                Op::Rng { stream: 0, shape: Shape::vector(32_000), kind: RngKind::Gumbel }, // 6
                Op::Add(5, 6),                // 7
                Op::ReduceArgmax(7),          // 8
            ],
            outputs: vec![OutputDecl::new(8, OutputKind::Token)],
        }
    }

    #[test]
    fn input_readiness_round_trip_and_recognizer_distinct() {
        // Slot 1 (a host mask) declared Late; slot 0 stays Submit (the default).
        let mut late = sample_program();
        late.inputs[1] = InputDecl::with_ready(Shape::SCALAR, DType::U32, Readiness::Late);
        let bytes = encode(&late);
        let back = decode(&bytes).expect("decode");
        assert_eq!(back.inputs[1].ready, Readiness::Late); // high-bit survives
        assert_eq!(back.inputs[1].dtype, DType::U32); // low 7 bits intact
        assert_eq!(back.inputs[0].ready, Readiness::Submit); // Submit default round-trips
        assert_eq!(back, late); // full identity
        // A Late-input program hashes distinctly — readiness rides the bytecode
        // `program_hash` hashes over, so it's a distinct recognized shape.
        assert_ne!(
            crate::program_hash(&bytes),
            crate::program_hash(&encode(&sample_program()))
        );
    }

    #[test]
    fn header_layout() {
        let p = sample_program();
        let b = encode(&p);
        assert_eq!(&b[0..4], b"PSIR");
        assert_eq!(u16::from_le_bytes([b[4], b[5]]), VERSION);
        assert_eq!(u16::from_le_bytes([b[6], b[7]]), 0); // flags
        assert_eq!(u32::from_le_bytes([b[8], b[9], b[10], b[11]]), 2); // n_inputs
        assert_eq!(u32::from_le_bytes([b[12], b[13], b[14], b[15]]), 9); // n_ops
        assert_eq!(u32::from_le_bytes([b[16], b[17], b[18], b[19]]), 1); // n_outputs
    }

    #[test]
    fn version_is_v4() {
        assert_eq!(u16::from_le_bytes([encode(&sample_program())[4], encode(&sample_program())[5]]), 4);
    }

    #[test]
    fn round_trip() {
        let p = sample_program();
        let bytes = encode(&p);
        assert_eq!(decode(&bytes).expect("decode"), p);
        assert_eq!(bytes, encode(&decode(&bytes).unwrap()));
    }

    #[test]
    fn round_trip_all_ops() {
        let ops = vec![
            Op::Input(0), Op::Input(1), Op::Const(Literal::F32(1.0)),
            Op::Exp(0), Op::Log(0), Op::Neg(0), Op::Recip(0), Op::Abs(0), Op::Sign(0),
            Op::Add(0, 0), Op::Sub(0, 0), Op::Mul(0, 0), Op::Div(0, 0),
            Op::MaxElem(0, 0), Op::MinElem(0, 0), Op::Gt(0, 0), Op::Ge(0, 0), Op::Eq(0, 0),
            Op::Select { cond: 0, a: 0, b: 0 },
            Op::ReduceSum(0), Op::ReduceMax(0), Op::ReduceMin(0), Op::ReduceArgmax(0),
            Op::Broadcast { value: 2, shape: Shape::matrix(2, 4) },
            Op::CumSum(0), Op::CumProd(0),
            Op::SortDesc(3),
            Op::PivotThreshold { input: 0, predicate: Predicate::RankLe(40) },
            Op::PivotThreshold { input: 0, predicate: Predicate::CummassLe(2) },
            Op::PivotThreshold { input: 0, predicate: Predicate::ProbGe(2) },
            Op::Gather { src: 3, idx: 1 },
            Op::GatherRow { src: 0, idx: 1 },
            Op::ScatterAdd { base: 3, idx: 1, vals: 1 },
            Op::ScatterSet { base: 3, idx: 1, vals: 1 },
            Op::Rng { stream: 1, shape: Shape::matrix(2, 4), kind: RngKind::Uniform },
            Op::Rng { stream: 0, shape: Shape::vector(4), kind: RngKind::Gumbel },
        ];
        let p = SamplingProgram {
            inputs: vec![
                InputDecl::new(Shape::matrix(2, 4), DType::F32),
                InputDecl::new(Shape::vector(4), DType::I32),
            ],
            ops,
            outputs: vec![OutputDecl::new(0, OutputKind::Logits)],
        };
        assert_eq!(decode(&encode(&p)).unwrap(), p);
    }

    #[test]
    fn shape_ranks_round_trip() {
        for shape in [Shape::SCALAR, Shape::vector(7), Shape::matrix(3, 5), Shape::new(&[2, 3, 4]).unwrap()] {
            let p = SamplingProgram {
                inputs: vec![],
                ops: vec![Op::Const(Literal::F32(1.0)), Op::Broadcast { value: 0, shape }],
                outputs: vec![OutputDecl::new(1, OutputKind::Logits)],
            };
            assert_eq!(decode(&encode(&p)).unwrap(), p);
        }
    }

    #[test]
    fn old_versions_rejected() {
        for v in [1u16, 2, 3, 5, 99] {
            let mut b = encode(&sample_program());
            b[4] = v as u8;
            b[5] = (v >> 8) as u8;
            assert_eq!(decode(&b), Err(DecodeError::UnsupportedVersion(v)));
        }
    }

    #[test]
    fn bad_magic_and_truncation() {
        let mut b = encode(&sample_program());
        b[0] = b'X';
        assert_eq!(decode(&b), Err(DecodeError::BadMagic));
        let b = encode(&sample_program());
        assert_eq!(decode(&b[..10]), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn decode_validated_catches_bad_ir() {
        let p = SamplingProgram {
            inputs: vec![InputDecl::new(Shape::vector(4), DType::F32)],
            ops: vec![Op::Input(0), Op::Exp(5)], // forward ref
            outputs: vec![OutputDecl::new(1, OutputKind::Logits)],
        };
        match decode_validated(&encode(&p)) {
            Err(DecodeErrorOrInvalid::Invalid(_)) => {}
            other => panic!("expected Invalid, got {other:?}"),
        }
    }
}
