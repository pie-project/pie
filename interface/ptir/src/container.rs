//! The PTIR **trace container** — the versioned blob carrying one traced
//! pass: stage-tagged programs, channel declarations, descriptor-port
//! bindings, and the name table for second-party kernels/sinks. Byte-for-byte
//! layout in `PTIR-CONTAINER.md` (the C++ driver reads that document).
//!
//! Identity = [`crate::container_hash`] (FNV-1a 64) over these canonical
//! bytes (contract C3). Canonical means: same trace ⟺ same bytes — the
//! encoder emits deterministically and the validator enforces the sortedness
//! rules (§2 of the doc), so the hash is a sound compile-cache / batching key.
//!
//! **Not in the container** (per-instance data, D2): channel seed *values*,
//! working-set binding, rng seeds. A seeded channel is declared `seeded = 1`
//! and its value arrives at instantiation.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::op::{ChannelIndex, IntrinsicId, Op};
use super::registry::{Port, Stage};
use crate::types::{DType, Literal, Predicate, RngKind, Shape, MAX_RANK};
use crate::{PTIR_MAGIC, PTIR_VERSION, PTIR_VERSION_EXTERN};

/// Channel element dtype: a concrete scalar type or the late-bound
/// model-intrinsic activation type (`ACT`, wire tag 4). `ACT` resolves to the
/// backend's quantized float at bind; in-program it materializes as F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChanDType {
    Concrete(DType),
    Act,
}

/// Wire tag for [`ChanDType::Act`].
pub const DT_ACT: u8 = 4;

impl ChanDType {
    pub fn tag(self) -> u8 {
        match self {
            ChanDType::Concrete(d) => d as u8,
            ChanDType::Act => DT_ACT,
        }
    }
    pub fn from_tag(t: u8) -> Option<Self> {
        Some(match t {
            0 => ChanDType::Concrete(DType::F32),
            1 => ChanDType::Concrete(DType::I32),
            2 => ChanDType::Concrete(DType::U32),
            3 => ChanDType::Concrete(DType::Bool),
            DT_ACT => ChanDType::Act,
            _ => return None,
        })
    }
    /// The dtype a program-side `take`/`read` of this channel yields (`ACT`
    /// materializes F32).
    pub fn program_dtype(self) -> DType {
        match self {
            ChanDType::Concrete(d) => d,
            ChanDType::Act => DType::F32,
        }
    }
}

/// The host endpoint of a channel, if any (the other endpoint is the pass).
/// SPSC (T2): `Writer` forbids any stage put; `Reader` forbids any stage
/// take/read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HostRole {
    None = 0,
    /// Host puts, pass consumes (e.g. §3's `mask`).
    Writer = 1,
    /// Pass puts, host takes/reads (e.g. §3's `out`).
    Reader = 2,
}

impl HostRole {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => HostRole::None,
            1 => HostRole::Writer,
            2 => HostRole::Reader,
            _ => return None,
        })
    }
}

/// One channel declaration (overview §1): GPU-resident ordered memory —
/// a bounded queue of cells with full/empty bits. Capacity is trace-known;
/// a capacity-N channel lowers to a ring of N+1 cells (§7.1).
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelDecl {
    pub shape: Shape,
    pub dtype: ChanDType,
    /// Queue capacity ≥ 1 (deeper run-ahead = larger capacity, §3).
    pub capacity: u32,
    pub host_role: HostRole,
    /// `Channel::from(v)`: starts full. The seed *value* is per-instance
    /// data supplied at instantiation — never in the container (D2).
    pub seeded: bool,
}

/// A descriptor port's source: a channel (contents read at execution time —
/// contract C1) or a trace-known constant (folded, e.g. a rectangular
/// `indptr`).
#[derive(Clone, Debug, PartialEq)]
pub enum PortSource {
    Channel(ChannelIndex),
    /// Raw little-endian payload: 4 bytes/element for F32/I32/U32, 1
    /// byte/element for Bool (the packed wire format is the runtime's, D1).
    Const { dtype: DType, shape: Shape, data: Vec<u8> },
}

/// One descriptor-port binding (overview §5.1).
#[derive(Clone, Debug, PartialEq)]
pub struct PortBinding {
    pub port: Port,
    pub source: PortSource,
}

/// Direction of an extern channel — whose endpoint THIS trace holds.
/// (v1.1 / wire-version 2; realizes §1's "SPSC pairs may span pipelines".)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExternDir {
    /// This trace CONSUMES: the other instance is the producer (e.g. the
    /// expert importing the amateur's logits channel). Stages may
    /// take/read, never put.
    Import = 0,
    /// This trace PRODUCES: the other instance consumes (e.g. the amateur
    /// exporting its logits). Stages may put, never take/read.
    Export = 1,
}

impl ExternDir {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => ExternDir::Import,
            1 => ExternDir::Export,
            _ => return None,
        })
    }
}

/// An extern-channel binding (v1.1): channel `chan`'s OTHER endpoint lives in
/// a different instance, paired at instantiation by `name` (an entry in the
/// container's name table). The channel decl itself keeps `host_role = None`
/// and `seeded = false` (the producer fills it); dtype/shape/capacity must
/// match the peer's at pairing time.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternDecl {
    pub name: crate::op::NameIndex,
    pub dir: ExternDir,
    pub chan: ChannelIndex,
}

/// One stage-tagged program: a flat SSA op list (see [`super::op`]).
#[derive(Clone, Debug, PartialEq)]
pub struct StageProgram {
    pub stage: Stage,
    pub ops: Vec<Op>,
}

/// A complete traced pass.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TraceContainer {
    /// Second-party kernel/sink names ([`Op::KernelCall`]/[`Op::SinkCall`]
    /// reference by index). Sorted + deduped for canonicality.
    pub names: Vec<String>,
    pub channels: Vec<ChannelDecl>,
    /// Sorted by port tag, unique.
    pub ports: Vec<PortBinding>,
    /// Sorted by stage tag, unique (at most one program per stage).
    pub stages: Vec<StageProgram>,
    /// v1.1 extern channels (sorted by `chan`, unique). When EMPTY the
    /// container encodes as wire-version 1 byte-identically (existing hashes
    /// never move); when present it encodes as version 2.
    pub externs: Vec<ExternDecl>,
}

impl TraceContainer {
    pub fn encode(&self) -> Vec<u8> {
        encode(self)
    }
    pub fn hash(&self) -> u64 {
        super::container_hash(&encode(self))
    }
}

// ===========================================================================
// Encode
// ===========================================================================

/// Lower a [`TraceContainer`] to its canonical bytes. Does not validate.
pub fn encode(c: &TraceContainer) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(&PTIR_MAGIC);
    // Wire-version selection keeps every pre-v1.1 hash stable: no externs ⇒
    // the exact version-1 byte stream; externs ⇒ version 2 (header gains
    // `n_externs`, table appended after the stages).
    let v2 = !c.externs.is_empty();
    put_u16(&mut w, if v2 { PTIR_VERSION_EXTERN } else { PTIR_VERSION });
    put_u16(&mut w, 0); // flags
    put_u32(&mut w, c.names.len() as u32);
    put_u32(&mut w, c.channels.len() as u32);
    put_u32(&mut w, c.ports.len() as u32);
    put_u32(&mut w, c.stages.len() as u32);
    if v2 {
        put_u32(&mut w, c.externs.len() as u32);
    }
    for n in &c.names {
        put_u16(&mut w, n.len() as u16);
        w.extend_from_slice(n.as_bytes());
    }
    for ch in &c.channels {
        w.push(ch.dtype.tag());
        encode_shape(&mut w, ch.shape);
        put_u32(&mut w, ch.capacity);
        w.push(ch.host_role as u8);
        w.push(ch.seeded as u8);
    }
    for p in &c.ports {
        w.push(p.port as u8);
        match &p.source {
            PortSource::Channel(ci) => {
                w.push(0);
                put_u32(&mut w, *ci);
            }
            PortSource::Const { dtype, shape, data } => {
                w.push(1);
                w.push(*dtype as u8);
                encode_shape(&mut w, *shape);
                w.extend_from_slice(data);
            }
        }
    }
    for s in &c.stages {
        w.push(s.stage as u8);
        put_u32(&mut w, s.ops.len() as u32);
        for op in &s.ops {
            encode_op(&mut w, op);
        }
    }
    for e in &c.externs {
        put_u16(&mut w, e.name);
        w.push(e.dir as u8);
        put_u32(&mut w, e.chan);
    }
    w
}

fn encode_op(w: &mut Vec<u8>, op: &Op) {
    w.push(op.tag());
    match *op {
        Op::Const(lit) => encode_literal(w, lit),

        Op::Exp(a) | Op::Log(a) | Op::Neg(a) | Op::Recip(a) | Op::Abs(a) | Op::Sign(a)
        | Op::Not(a) | Op::ReduceSum(a) | Op::ReduceMax(a) | Op::ReduceMin(a)
        | Op::ReduceArgmax(a) | Op::Transpose(a) | Op::CumSum(a) | Op::CumProd(a)
        | Op::SortDesc(a) => put_u32(w, a),

        Op::Cast { value, dtype } => {
            put_u32(w, value);
            w.push(dtype as u8);
        }

        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::MaxElem(a, b)
        | Op::MinElem(a, b) | Op::Rem(a, b) | Op::Gt(a, b) | Op::Ge(a, b) | Op::Eq(a, b)
        | Op::Ne(a, b) | Op::Lt(a, b) | Op::Le(a, b) | Op::And(a, b) | Op::Or(a, b)
        | Op::MatMul(a, b) => {
            put_u32(w, a);
            put_u32(w, b);
        }

        Op::Select { cond, a, b } => {
            put_u32(w, cond);
            put_u32(w, a);
            put_u32(w, b);
        }

        Op::Broadcast { value, shape } | Op::Reshape { value, shape } => {
            put_u32(w, value);
            encode_shape(w, shape);
        }

        Op::TopK { input, k } => {
            put_u32(w, input);
            put_u32(w, k);
        }

        Op::PivotThreshold { input, predicate } => {
            put_u32(w, input);
            encode_predicate(w, predicate);
        }

        Op::Gather { src, idx } | Op::GatherRow { src, idx } => {
            put_u32(w, src);
            put_u32(w, idx);
        }
        Op::MaskApply { logits, mask } => {
            put_u32(w, logits);
            put_u32(w, mask);
        }
        Op::ScatterAdd { base, idx, vals } | Op::ScatterSet { base, idx, vals } => {
            put_u32(w, base);
            put_u32(w, idx);
            put_u32(w, vals);
        }
        Op::Iota { len } => put_u32(w, len),

        Op::Rng { stream, shape, kind } => {
            put_u32(w, stream);
            encode_shape(w, shape);
            w.push(kind as u8);
        }
        Op::RngKeyed { state, shape, kind } => {
            put_u32(w, state);
            encode_shape(w, shape);
            w.push(kind as u8);
        }

        Op::ChanTake(c) | Op::ChanRead(c) => put_u32(w, c),
        Op::ChanPut { chan, value } => {
            put_u32(w, chan);
            put_u32(w, value);
        }

        Op::IntrinsicVal { intr, shape, dtype } => {
            put_u16(w, intr as u16);
            w.push(dtype as u8);
            encode_shape(w, shape);
        }
        Op::KernelCall { name, ref args, shape, dtype } => {
            put_u16(w, name);
            w.push(dtype as u8);
            encode_shape(w, shape);
            w.push(args.len() as u8);
            for &a in args {
                put_u32(w, a);
            }
        }
        Op::SinkCall { name, ref args } => {
            put_u16(w, name);
            w.push(args.len() as u8);
            for &a in args {
                put_u32(w, a);
            }
        }
    }
}

fn encode_predicate(w: &mut Vec<u8>, pred: Predicate) {
    match pred {
        Predicate::RankLe(v) => {
            w.push(0);
            put_u32(w, v);
        }
        Predicate::CummassLe(v) => {
            w.push(1);
            put_u32(w, v);
        }
        Predicate::ProbGe(v) => {
            w.push(2);
            put_u32(w, v);
        }
    }
}

fn encode_shape(w: &mut Vec<u8>, shape: Shape) {
    w.push(shape.rank() as u8);
    for &d in shape.dims() {
        put_u32(w, d);
    }
}

fn encode_literal(w: &mut Vec<u8>, lit: Literal) {
    match lit {
        Literal::F32(x) => {
            w.push(0);
            put_u32(w, x.to_bits());
        }
        Literal::I32(x) => {
            w.push(1);
            put_u32(w, x as u32);
        }
        Literal::U32(x) => {
            w.push(2);
            put_u32(w, x);
        }
        Literal::Bool(b) => {
            w.push(3);
            put_u32(w, b as u32);
        }
    }
}

fn put_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

/// Bytes per element of a const-port payload.
pub fn const_elem_size(dtype: DType) -> usize {
    match dtype {
        DType::Bool => 1,
        _ => 4,
    }
}

// ===========================================================================
// Decode
// ===========================================================================

/// A container decode failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContainerDecodeError {
    BadMagic,
    UnsupportedVersion(u16),
    UnexpectedEof,
    UnknownOpcode(u8),
    UnknownTag { what: &'static str, tag: u8 },
    RankTooLarge(u8),
    BadUtf8,
}

impl fmt::Display for ContainerDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ContainerDecodeError::*;
        match self {
            BadMagic => f.write_str("bad magic (expected \"PTIR\")"),
            UnsupportedVersion(v) => write!(f, "unsupported container version {v}"),
            UnexpectedEof => f.write_str("unexpected end of buffer"),
            UnknownOpcode(t) => write!(f, "unknown opcode 0x{t:02x}"),
            UnknownTag { what, tag } => write!(f, "unknown {what} tag 0x{tag:02x}"),
            RankTooLarge(r) => write!(f, "shape rank {r} exceeds MAX_RANK"),
            BadUtf8 => f.write_str("name table entry is not valid UTF-8"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ContainerDecodeError {}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8, ContainerDecodeError> {
        let b = *self.buf.get(self.pos).ok_or(ContainerDecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }
    fn u16(&mut self) -> Result<u16, ContainerDecodeError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32, ContainerDecodeError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], ContainerDecodeError> {
        let end = self.pos.checked_add(n).ok_or(ContainerDecodeError::UnexpectedEof)?;
        let s = self.buf.get(self.pos..end).ok_or(ContainerDecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(s)
    }
}

/// Parse container bytes back into the model. Does not validate (bind does).
pub fn decode(bytes: &[u8]) -> Result<TraceContainer, ContainerDecodeError> {
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.take(4)? != PTIR_MAGIC {
        return Err(ContainerDecodeError::BadMagic);
    }
    let version = r.u16()?;
    if version != PTIR_VERSION && version != PTIR_VERSION_EXTERN {
        return Err(ContainerDecodeError::UnsupportedVersion(version));
    }
    let _flags = r.u16()?;
    let n_names = r.u32()?;
    let n_channels = r.u32()?;
    let n_ports = r.u32()?;
    let n_stages = r.u32()?;
    let n_externs = if version == PTIR_VERSION_EXTERN { r.u32()? } else { 0 };

    let mut names = Vec::with_capacity(n_names as usize);
    for _ in 0..n_names {
        let len = r.u16()? as usize;
        let bytes = r.take(len)?;
        names.push(String::from_utf8(bytes.to_vec()).map_err(|_| ContainerDecodeError::BadUtf8)?);
    }

    let mut channels = Vec::with_capacity(n_channels as usize);
    for _ in 0..n_channels {
        let dt = r.u8()?;
        let dtype = ChanDType::from_tag(dt)
            .ok_or(ContainerDecodeError::UnknownTag { what: "channel dtype", tag: dt })?;
        let shape = decode_shape(&mut r)?;
        let capacity = r.u32()?;
        let hr = r.u8()?;
        let host_role = HostRole::from_u8(hr)
            .ok_or(ContainerDecodeError::UnknownTag { what: "host role", tag: hr })?;
        let seeded = r.u8()? != 0;
        channels.push(ChannelDecl { shape, dtype, capacity, host_role, seeded });
    }

    let mut ports = Vec::with_capacity(n_ports as usize);
    for _ in 0..n_ports {
        let pt = r.u8()?;
        let port =
            Port::from_u8(pt).ok_or(ContainerDecodeError::UnknownTag { what: "port", tag: pt })?;
        let src = r.u8()?;
        let source = match src {
            0 => PortSource::Channel(r.u32()?),
            1 => {
                let dt = r.u8()?;
                let dtype = decode_dtype(dt)?;
                let shape = decode_shape(&mut r)?;
                let n = shape.numel() as usize * const_elem_size(dtype);
                PortSource::Const { dtype, shape, data: r.take(n)?.to_vec() }
            }
            t => return Err(ContainerDecodeError::UnknownTag { what: "port source", tag: t }),
        };
        ports.push(PortBinding { port, source });
    }

    let mut stages = Vec::with_capacity(n_stages as usize);
    for _ in 0..n_stages {
        let st = r.u8()?;
        let stage =
            Stage::from_u8(st).ok_or(ContainerDecodeError::UnknownTag { what: "stage", tag: st })?;
        let n_ops = r.u32()?;
        let mut ops = Vec::with_capacity(n_ops as usize);
        for _ in 0..n_ops {
            ops.push(decode_op(&mut r)?);
        }
        stages.push(StageProgram { stage, ops });
    }
    let mut externs = Vec::with_capacity(n_externs as usize);
    for _ in 0..n_externs {
        let name = r.u16()?;
        let d = r.u8()?;
        let dir = ExternDir::from_u8(d)
            .ok_or(ContainerDecodeError::UnknownTag { what: "extern dir", tag: d })?;
        let chan = r.u32()?;
        externs.push(ExternDecl { name, dir, chan });
    }
    Ok(TraceContainer { names, channels, ports, stages, externs })
}

fn decode_op(r: &mut Reader<'_>) -> Result<Op, ContainerDecodeError> {
    let tag = r.u8()?;
    let op = match tag {
        0x01 => Op::Exp(r.u32()?),
        0x02 => Op::Log(r.u32()?),
        0x03 => Op::Neg(r.u32()?),
        0x04 => Op::Recip(r.u32()?),
        0x05 => Op::Abs(r.u32()?),
        0x06 => Op::Sign(r.u32()?),
        0x07 => Op::Cast { value: r.u32()?, dtype: decode_dtype(r.u8()?)? },
        0x10 => Op::Add(r.u32()?, r.u32()?),
        0x11 => Op::Sub(r.u32()?, r.u32()?),
        0x12 => Op::Mul(r.u32()?, r.u32()?),
        0x13 => Op::Div(r.u32()?, r.u32()?),
        0x14 => Op::MaxElem(r.u32()?, r.u32()?),
        0x15 => Op::MinElem(r.u32()?, r.u32()?),
        0x16 => Op::Gt(r.u32()?, r.u32()?),
        0x17 => Op::Ge(r.u32()?, r.u32()?),
        0x18 => Op::Eq(r.u32()?, r.u32()?),
        0x19 => Op::Ne(r.u32()?, r.u32()?),
        0x1A => Op::Lt(r.u32()?, r.u32()?),
        0x1B => Op::Le(r.u32()?, r.u32()?),
        0x1C => Op::And(r.u32()?, r.u32()?),
        0x1D => Op::Or(r.u32()?, r.u32()?),
        0x1E => Op::Not(r.u32()?),
        0x1F => Op::Rem(r.u32()?, r.u32()?),
        0x20 => Op::Select { cond: r.u32()?, a: r.u32()?, b: r.u32()? },
        0x30 => Op::ReduceSum(r.u32()?),
        0x31 => Op::ReduceMax(r.u32()?),
        0x32 => Op::ReduceMin(r.u32()?),
        0x33 => Op::ReduceArgmax(r.u32()?),
        0x38 => Op::Broadcast { value: r.u32()?, shape: decode_shape(r)? },
        0x39 => Op::Reshape { value: r.u32()?, shape: decode_shape(r)? },
        0x3A => Op::Transpose(r.u32()?),
        0x40 => Op::CumSum(r.u32()?),
        0x41 => Op::CumProd(r.u32()?),
        0x50 => Op::SortDesc(r.u32()?),
        0x51 => Op::TopK { input: r.u32()?, k: r.u32()? },
        0x55 => Op::MatMul(r.u32()?, r.u32()?),
        0x58 => {
            let input = r.u32()?;
            let predicate = match r.u8()? {
                0 => Predicate::RankLe(r.u32()?),
                1 => Predicate::CummassLe(r.u32()?),
                2 => Predicate::ProbGe(r.u32()?),
                t => return Err(ContainerDecodeError::UnknownTag { what: "predicate", tag: t }),
            };
            Op::PivotThreshold { input, predicate }
        }
        0x60 => Op::Gather { src: r.u32()?, idx: r.u32()? },
        0x61 => Op::GatherRow { src: r.u32()?, idx: r.u32()? },
        0x62 => Op::ScatterAdd { base: r.u32()?, idx: r.u32()?, vals: r.u32()? },
        0x63 => Op::ScatterSet { base: r.u32()?, idx: r.u32()?, vals: r.u32()? },
        0x64 => Op::Iota { len: r.u32()? },
        0x65 => Op::MaskApply { logits: r.u32()?, mask: r.u32()? },
        0x70 => Op::Rng { stream: r.u32()?, shape: decode_shape(r)?, kind: decode_rng_kind(r.u8()?)? },
        0x71 => Op::RngKeyed { state: r.u32()?, shape: decode_shape(r)?, kind: decode_rng_kind(r.u8()?)? },
        0x81 => {
            let dt = r.u8()?;
            let bits = r.u32()?;
            Op::Const(match dt {
                0 => Literal::F32(f32::from_bits(bits)),
                1 => Literal::I32(bits as i32),
                2 => Literal::U32(bits),
                3 => Literal::Bool(bits != 0),
                t => return Err(ContainerDecodeError::UnknownTag { what: "literal dtype", tag: t }),
            })
        }
        0x90 => Op::ChanTake(r.u32()?),
        0x91 => Op::ChanRead(r.u32()?),
        0x92 => Op::ChanPut { chan: r.u32()?, value: r.u32()? },
        0xA0 => {
            let iv = r.u16()?;
            let intr = IntrinsicId::from_u16(iv).ok_or(ContainerDecodeError::UnknownTag {
                what: "intrinsic",
                tag: iv as u8,
            })?;
            let dtype = decode_dtype(r.u8()?)?;
            let shape = decode_shape(r)?;
            Op::IntrinsicVal { intr, shape, dtype }
        }
        0xA1 => {
            let name = r.u16()?;
            let dtype = decode_dtype(r.u8()?)?;
            let shape = decode_shape(r)?;
            let n = r.u8()? as usize;
            let mut args = Vec::with_capacity(n);
            for _ in 0..n {
                args.push(r.u32()?);
            }
            Op::KernelCall { name, args, shape, dtype }
        }
        0xA2 => {
            let name = r.u16()?;
            let n = r.u8()? as usize;
            let mut args = Vec::with_capacity(n);
            for _ in 0..n {
                args.push(r.u32()?);
            }
            Op::SinkCall { name, args }
        }
        t => return Err(ContainerDecodeError::UnknownOpcode(t)),
    };
    Ok(op)
}

fn decode_rng_kind(t: u8) -> Result<RngKind, ContainerDecodeError> {
    Ok(match t {
        0 => RngKind::Uniform,
        1 => RngKind::Gumbel,
        t => return Err(ContainerDecodeError::UnknownTag { what: "rng kind", tag: t }),
    })
}

fn decode_dtype(t: u8) -> Result<DType, ContainerDecodeError> {
    Ok(match t {
        0 => DType::F32,
        1 => DType::I32,
        2 => DType::U32,
        3 => DType::Bool,
        t => return Err(ContainerDecodeError::UnknownTag { what: "dtype", tag: t }),
    })
}

fn decode_shape(r: &mut Reader<'_>) -> Result<Shape, ContainerDecodeError> {
    let rank = r.u8()?;
    if rank as usize > MAX_RANK {
        return Err(ContainerDecodeError::RankTooLarge(rank));
    }
    let mut dims = [0u32; MAX_RANK];
    for d in dims.iter_mut().take(rank as usize) {
        *d = r.u32()?;
    }
    Ok(Shape::new(&dims[..rank as usize]).expect("rank checked"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn sample() -> TraceContainer {
        // A miniature two-channel greedy epilogue: tok (device loop-carried),
        // out (host-read); epilogue = argmax(logits) -> tok, out.
        let vocab = 32u32;
        TraceContainer {
            names: vec!["envelope_dot".to_string()],
            channels: vec![
                ChannelDecl {
                    shape: Shape::vector(1),
                    dtype: ChanDType::Concrete(DType::I32),
                    capacity: 1,
                    host_role: HostRole::None,
                    seeded: true,
                },
                ChannelDecl {
                    shape: Shape::vector(1),
                    dtype: ChanDType::Concrete(DType::I32),
                    capacity: 1,
                    host_role: HostRole::Reader,
                    seeded: false,
                },
            ],
            ports: vec![
                PortBinding { port: Port::EmbedTokens, source: PortSource::Channel(0) },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    },
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Logits,
                        shape: Shape::matrix(1, vocab),
                        dtype: DType::F32,
                    }, // id 0
                    Op::ReduceArgmax(0), // id 1
                    Op::ChanPut { chan: 0, value: 1 },
                    Op::ChanPut { chan: 1, value: 1 },
                ],
            }],
            externs: Vec::new(),
        }
    }

    #[test]
    fn round_trip() {
        let c = sample();
        let bytes = encode(&c);
        assert_eq!(decode(&bytes).expect("decode"), c);
        assert_eq!(bytes, encode(&decode(&bytes).unwrap()));
    }

    #[test]
    fn round_trip_2d_channel_shape() {
        // Regression: a §6.2 beam `pages` channel is [B,P] (2D). The container
        // encode/decode MUST preserve the 2D shape (numel B*P), else validate_seeds
        // rejects the [B,P] seed as a byte-length mismatch.
        let mut c = sample();
        c.channels[0].shape = Shape::matrix(2, 4);
        let bytes = encode(&c);
        let d = decode(&bytes).expect("decode");
        assert_eq!(d.channels[0].shape.dims(), &[2, 4], "2D dims must survive");
        assert_eq!(d.channels[0].shape.numel(), 8, "2D [2,4] numel must be 8");
    }

    #[test]
    fn hash_is_stable_and_seed_independent() {
        let c = sample();
        assert_eq!(c.hash(), c.hash());
        // Identity ignores nothing in the bytes — but seeds are not IN the
        // bytes, so two instances differing only in seed values share one
        // identity by construction.
        let mut c2 = sample();
        c2.channels[0].seeded = false; // structural change ⇒ different identity
        assert_ne!(c.hash(), c2.hash());
    }

    #[test]
    fn round_trip_every_op() {
        let ops = vec![
            Op::Const(Literal::F32(0.5)),
            Op::Const(Literal::I32(-1)),
            Op::Const(Literal::U32(7)),
            Op::Const(Literal::Bool(true)),
            Op::Exp(0), Op::Log(0), Op::Neg(0), Op::Recip(0), Op::Abs(0), Op::Sign(0),
            Op::Cast { value: 0, dtype: DType::U32 },
            Op::Add(0, 1), Op::Sub(0, 1), Op::Mul(0, 1), Op::Div(0, 1),
            Op::MaxElem(0, 1), Op::MinElem(0, 1), Op::Rem(0, 1),
            Op::Gt(0, 1), Op::Ge(0, 1), Op::Eq(0, 1), Op::Ne(0, 1), Op::Lt(0, 1), Op::Le(0, 1),
            Op::And(4, 5), Op::Or(4, 5), Op::Not(4),
            Op::Select { cond: 4, a: 0, b: 1 },
            Op::ReduceSum(0), Op::ReduceMax(0), Op::ReduceMin(0), Op::ReduceArgmax(0),
            Op::Broadcast { value: 0, shape: Shape::matrix(2, 3) },
            Op::Reshape { value: 0, shape: Shape::vector(6) },
            Op::Transpose(0),
            Op::CumSum(0), Op::CumProd(0),
            Op::SortDesc(0),
            Op::TopK { input: 0, k: 3 },
            Op::MatMul(0, 1),
            Op::PivotThreshold { input: 0, predicate: Predicate::CummassLe(2) },
            Op::Gather { src: 0, idx: 1 },
            Op::GatherRow { src: 0, idx: 1 },
            Op::ScatterAdd { base: 0, idx: 1, vals: 2 },
            Op::ScatterSet { base: 0, idx: 1, vals: 2 },
            Op::Iota { len: 5 },
            Op::MaskApply { logits: 0, mask: 1 },
            Op::Rng { stream: 2, shape: Shape::vector(4), kind: RngKind::Uniform },
            Op::RngKeyed { state: 0, shape: Shape::vector(4), kind: RngKind::Gumbel },
            Op::ChanTake(0),
            Op::ChanRead(1),
            Op::ChanPut { chan: 0, value: 0 },
            Op::IntrinsicVal { intr: IntrinsicId::Layer, shape: Shape::SCALAR, dtype: DType::U32 },
            Op::KernelCall { name: 0, args: vec![0, 1, 2], shape: Shape::vector(9), dtype: DType::F32 },
            Op::SinkCall { name: 0, args: vec![0] },
        ];
        let c = TraceContainer {
            names: vec!["k".to_string()],
            channels: vec![
                ChannelDecl {
                    shape: Shape::vector(4),
                    dtype: ChanDType::Act,
                    capacity: 2,
                    host_role: HostRole::Writer,
                    seeded: false,
                },
                ChannelDecl {
                    shape: Shape::SCALAR,
                    dtype: ChanDType::Concrete(DType::F32),
                    capacity: 1,
                    host_role: HostRole::None,
                    seeded: true,
                },
            ],
            ports: vec![],
            stages: vec![StageProgram { stage: Stage::Prologue, ops }],
        externs: alloc::vec::Vec::new(),
        };
        let bytes = encode(&c);
        assert_eq!(decode(&bytes).expect("decode"), c);
    }

    #[test]
    fn rejects_bad_magic_version_and_truncation() {
        let mut b = encode(&sample());
        b[0] = b'X';
        assert_eq!(decode(&b), Err(ContainerDecodeError::BadMagic));
        let mut b = encode(&sample());
        b[4] = 9;
        assert_eq!(decode(&b), Err(ContainerDecodeError::UnsupportedVersion(9)));
        let b = encode(&sample());
        assert_eq!(decode(&b[..b.len() - 2]), Err(ContainerDecodeError::UnexpectedEof));
    }
}
