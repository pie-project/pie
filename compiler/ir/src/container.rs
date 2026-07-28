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

use super::op::{ChannelIndex, IntrinsicId, Op, tags};
use super::read::{ReadError, Reader};
use super::registry::{Port, Stage};
use crate::types::{DType, Literal, MAX_RANK, Predicate, RngKind, Shape};
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
    Const {
        dtype: DType,
        shape: Shape,
        data: Vec<u8>,
    },
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
    // Preserve version-1 hashes when the extern extension is absent.
    let v2 = !c.externs.is_empty();
    put_u16(
        &mut w,
        if v2 {
            PTIR_VERSION_EXTERN
        } else {
            PTIR_VERSION
        },
    );
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

pub fn encode_op(w: &mut Vec<u8>, op: &Op) {
    w.push(op.tag());
    match *op {
        Op::Const(lit) => encode_literal(w, lit),

        Op::Exp(a)
        | Op::Log(a)
        | Op::Neg(a)
        | Op::Recip(a)
        | Op::Abs(a)
        | Op::Sign(a)
        | Op::Not(a)
        | Op::ReduceSum(a)
        | Op::ReduceMax(a)
        | Op::ReduceMin(a)
        | Op::ReduceArgmax(a)
        | Op::Transpose(a)
        | Op::CumSum(a)
        | Op::CumProd(a)
        | Op::SortDesc(a) => put_u32(w, a),

        Op::Cast { value, dtype } => {
            put_u32(w, value);
            w.push(dtype as u8);
        }

        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::MaxElem(a, b)
        | Op::MinElem(a, b)
        | Op::Rem(a, b)
        | Op::Gt(a, b)
        | Op::Ge(a, b)
        | Op::Eq(a, b)
        | Op::Ne(a, b)
        | Op::Lt(a, b)
        | Op::Le(a, b)
        | Op::And(a, b)
        | Op::Or(a, b)
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
        Op::CausalMask { positions, len } => {
            put_u32(w, positions);
            put_u32(w, len);
        }
        Op::SlidingWindowMask {
            positions,
            len,
            window,
        } => {
            put_u32(w, positions);
            put_u32(w, len);
            put_u32(w, window);
        }
        Op::SinkWindowMask {
            positions,
            len,
            sink,
            window,
        } => {
            put_u32(w, positions);
            put_u32(w, len);
            put_u32(w, sink);
            put_u32(w, window);
        }
        Op::ScatterAdd { base, idx, vals } | Op::ScatterSet { base, idx, vals } => {
            put_u32(w, base);
            put_u32(w, idx);
            put_u32(w, vals);
        }
        Op::Iota { len } => put_u32(w, len),

        Op::Rng {
            stream,
            shape,
            kind,
        } => {
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
        Op::KernelCall {
            name,
            ref args,
            shape,
            dtype,
        } => {
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

pub fn encode_shape(w: &mut Vec<u8>, shape: Shape) {
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

pub fn put_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

/// Bytes per element of a const-port payload.
///
/// Exhaustive on purpose: a `_ => 4` arm answers "4 bytes" for an F16 or E8M0
/// dtype the day one is added, and the result is a mis-sized payload rather
/// than a compile error.
pub fn const_elem_size(dtype: DType) -> usize {
    match dtype {
        DType::Bool => 1,
        DType::F32 | DType::I32 | DType::U32 => 4,
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
    ZeroDimension,
    BadUtf8,
    TrailingBytes,
    NonCanonical,
    CountTooLarge(&'static str),
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
            ZeroDimension => f.write_str("shape dimensions must be nonzero"),
            BadUtf8 => f.write_str("name table entry is not valid UTF-8"),
            TrailingBytes => f.write_str("trailing bytes after PTIR container"),
            NonCanonical => f.write_str("noncanonical PTIR container encoding"),
            CountTooLarge(table) => write!(f, "{table} count exceeds remaining container bytes"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ContainerDecodeError {}

impl From<ReadError> for ContainerDecodeError {
    fn from(error: ReadError) -> Self {
        match error {
            ReadError::UnexpectedEof => ContainerDecodeError::UnexpectedEof,
            ReadError::CountTooLarge(table) => ContainerDecodeError::CountTooLarge(table),
        }
    }
}

/// Decoder ceilings.
///
/// These are resource limits, not semantic ones: a container within them can
/// still be rejected by `bind`, and the numbers are chosen to bound work and
/// memory for input we did not write, not to express what a sampling pass is
/// allowed to say. They sit far above anything a traced pass produces -- the
/// whole test corpus is three orders of magnitude below `MAX_OPS`.
///
/// `MAX_STAGES` is different in kind: a container carries at most one program
/// per stage, so `Stage::ALL.len()` is a structural fact rather than a budget.
pub const MAX_STAGES: usize = Stage::ALL.len();
/// Per-stage op ceiling. Planning is linear in this, so it bounds compile time
/// as well as memory.
pub const MAX_OPS: usize = 1 << 16;
pub const MAX_CHANNELS: usize = 1 << 12;
pub const MAX_NAMES: usize = 1 << 12;
pub const MAX_PORTS: usize = 1 << 8;
pub const MAX_EXTERNS: usize = MAX_CHANNELS;

/// Parse container bytes back into the model. Does not validate (bind does).
pub fn decode(bytes: &[u8]) -> Result<TraceContainer, ContainerDecodeError> {
    let mut r = Reader::new(bytes);
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
    let n_externs = if version == PTIR_VERSION_EXTERN {
        r.u32()?
    } else {
        0
    };

    let mut names = Vec::with_capacity(r.bounded_count(n_names, 2, MAX_NAMES, "name table")?);
    for _ in 0..n_names {
        let len = r.u16()? as usize;
        let bytes = r.take(len)?;
        names.push(String::from_utf8(bytes.to_vec()).map_err(|_| ContainerDecodeError::BadUtf8)?);
    }

    let mut channels =
        Vec::with_capacity(r.bounded_count(n_channels, 8, MAX_CHANNELS, "channel table")?);
    for _ in 0..n_channels {
        let dt = r.u8()?;
        let dtype = ChanDType::from_tag(dt).ok_or(ContainerDecodeError::UnknownTag {
            what: "channel dtype",
            tag: dt,
        })?;
        let shape = decode_shape(&mut r)?;
        let capacity = r.u32()?;
        let hr = r.u8()?;
        let host_role = HostRole::from_u8(hr).ok_or(ContainerDecodeError::UnknownTag {
            what: "host role",
            tag: hr,
        })?;
        let seeded = r.u8()? != 0;
        channels.push(ChannelDecl {
            shape,
            dtype,
            capacity,
            host_role,
            seeded,
        });
    }

    let mut ports = Vec::with_capacity(r.bounded_count(n_ports, 4, MAX_PORTS, "port table")?);
    for _ in 0..n_ports {
        let pt = r.u8()?;
        let port = Port::from_u8(pt).ok_or(ContainerDecodeError::UnknownTag {
            what: "port",
            tag: pt,
        })?;
        let src = r.u8()?;
        let source = match src {
            0 => PortSource::Channel(r.u32()?),
            1 => {
                let dt = r.u8()?;
                let dtype = decode_dtype(dt)?;
                let shape = decode_shape(&mut r)?;
                let n = usize::try_from(shape.numel())
                    .ok()
                    .and_then(|numel| numel.checked_mul(const_elem_size(dtype)))
                    .ok_or(ContainerDecodeError::CountTooLarge("port constant payload"))?;
                PortSource::Const {
                    dtype,
                    shape,
                    data: r.take(n)?.to_vec(),
                }
            }
            t => {
                return Err(ContainerDecodeError::UnknownTag {
                    what: "port source",
                    tag: t,
                });
            }
        };
        ports.push(PortBinding { port, source });
    }

    let mut stages = Vec::with_capacity(r.bounded_count(n_stages, 5, MAX_STAGES, "stage table")?);
    for _ in 0..n_stages {
        let st = r.u8()?;
        let stage = Stage::from_u8(st).ok_or(ContainerDecodeError::UnknownTag {
            what: "stage",
            tag: st,
        })?;
        let n_ops = r.u32()?;
        let mut ops = Vec::with_capacity(r.bounded_count(n_ops, 1, MAX_OPS, "operation table")?);
        for _ in 0..n_ops {
            ops.push(decode_op(&mut r)?);
        }
        stages.push(StageProgram { stage, ops });
    }
    let mut externs =
        Vec::with_capacity(r.bounded_count(n_externs, 7, MAX_EXTERNS, "extern table")?);
    for _ in 0..n_externs {
        let name = r.u16()?;
        let d = r.u8()?;
        let dir = ExternDir::from_u8(d).ok_or(ContainerDecodeError::UnknownTag {
            what: "extern dir",
            tag: d,
        })?;
        let chan = r.u32()?;
        externs.push(ExternDecl { name, dir, chan });
    }
    if r.offset() != bytes.len() {
        return Err(ContainerDecodeError::TrailingBytes);
    }
    let container = TraceContainer {
        names,
        channels,
        ports,
        stages,
        externs,
    };
    if container.encode() != bytes {
        return Err(ContainerDecodeError::NonCanonical);
    }
    Ok(container)
}

fn decode_op(r: &mut Reader<'_>) -> Result<Op, ContainerDecodeError> {
    let tag = r.u8()?;
    let op = match tag {
        tags::EXP => Op::Exp(r.u32()?),
        tags::LOG => Op::Log(r.u32()?),
        tags::NEG => Op::Neg(r.u32()?),
        tags::RECIP => Op::Recip(r.u32()?),
        tags::ABS => Op::Abs(r.u32()?),
        tags::SIGN => Op::Sign(r.u32()?),
        tags::CAST => Op::Cast {
            value: r.u32()?,
            dtype: decode_dtype(r.u8()?)?,
        },
        tags::ADD => Op::Add(r.u32()?, r.u32()?),
        tags::SUB => Op::Sub(r.u32()?, r.u32()?),
        tags::MUL => Op::Mul(r.u32()?, r.u32()?),
        tags::DIV => Op::Div(r.u32()?, r.u32()?),
        tags::MAX_ELEM => Op::MaxElem(r.u32()?, r.u32()?),
        tags::MIN_ELEM => Op::MinElem(r.u32()?, r.u32()?),
        tags::GT => Op::Gt(r.u32()?, r.u32()?),
        tags::GE => Op::Ge(r.u32()?, r.u32()?),
        tags::EQ => Op::Eq(r.u32()?, r.u32()?),
        tags::NE => Op::Ne(r.u32()?, r.u32()?),
        tags::LT => Op::Lt(r.u32()?, r.u32()?),
        tags::LE => Op::Le(r.u32()?, r.u32()?),
        tags::AND => Op::And(r.u32()?, r.u32()?),
        tags::OR => Op::Or(r.u32()?, r.u32()?),
        tags::NOT => Op::Not(r.u32()?),
        tags::REM => Op::Rem(r.u32()?, r.u32()?),
        tags::SELECT => Op::Select {
            cond: r.u32()?,
            a: r.u32()?,
            b: r.u32()?,
        },
        tags::REDUCE_SUM => Op::ReduceSum(r.u32()?),
        tags::REDUCE_MAX => Op::ReduceMax(r.u32()?),
        tags::REDUCE_MIN => Op::ReduceMin(r.u32()?),
        tags::REDUCE_ARGMAX => Op::ReduceArgmax(r.u32()?),
        tags::BROADCAST => Op::Broadcast {
            value: r.u32()?,
            shape: decode_shape(r)?,
        },
        tags::RESHAPE => Op::Reshape {
            value: r.u32()?,
            shape: decode_shape(r)?,
        },
        tags::TRANSPOSE => Op::Transpose(r.u32()?),
        tags::CUMSUM => Op::CumSum(r.u32()?),
        tags::CUMPROD => Op::CumProd(r.u32()?),
        tags::SORT_DESC => Op::SortDesc(r.u32()?),
        tags::TOP_K => Op::TopK {
            input: r.u32()?,
            k: r.u32()?,
        },
        tags::MATMUL => Op::MatMul(r.u32()?, r.u32()?),
        tags::PIVOT_THRESHOLD => {
            let input = r.u32()?;
            let predicate = match r.u8()? {
                0 => Predicate::RankLe(r.u32()?),
                1 => Predicate::CummassLe(r.u32()?),
                2 => Predicate::ProbGe(r.u32()?),
                t => {
                    return Err(ContainerDecodeError::UnknownTag {
                        what: "predicate",
                        tag: t,
                    });
                }
            };
            Op::PivotThreshold { input, predicate }
        }
        tags::GATHER => Op::Gather {
            src: r.u32()?,
            idx: r.u32()?,
        },
        tags::GATHER_ROW => Op::GatherRow {
            src: r.u32()?,
            idx: r.u32()?,
        },
        tags::SCATTER_ADD => Op::ScatterAdd {
            base: r.u32()?,
            idx: r.u32()?,
            vals: r.u32()?,
        },
        tags::SCATTER_SET => Op::ScatterSet {
            base: r.u32()?,
            idx: r.u32()?,
            vals: r.u32()?,
        },
        tags::IOTA => Op::Iota { len: r.u32()? },
        tags::MASK_APPLY_PACKED => Op::MaskApply {
            logits: r.u32()?,
            mask: r.u32()?,
        },
        tags::CAUSAL_MASK => Op::CausalMask {
            positions: r.u32()?,
            len: r.u32()?,
        },
        tags::SLIDING_WINDOW_MASK => Op::SlidingWindowMask {
            positions: r.u32()?,
            len: r.u32()?,
            window: r.u32()?,
        },
        tags::SINK_WINDOW_MASK => Op::SinkWindowMask {
            positions: r.u32()?,
            len: r.u32()?,
            sink: r.u32()?,
            window: r.u32()?,
        },
        tags::RNG => Op::Rng {
            stream: r.u32()?,
            shape: decode_shape(r)?,
            kind: decode_rng_kind(r.u8()?)?,
        },
        tags::RNG_KEYED => Op::RngKeyed {
            state: r.u32()?,
            shape: decode_shape(r)?,
            kind: decode_rng_kind(r.u8()?)?,
        },
        tags::CONST => {
            let dt = r.u8()?;
            let bits = r.u32()?;
            Op::Const(match dt {
                0 => Literal::F32(f32::from_bits(bits)),
                1 => Literal::I32(bits as i32),
                2 => Literal::U32(bits),
                3 => Literal::Bool(bits != 0),
                t => {
                    return Err(ContainerDecodeError::UnknownTag {
                        what: "literal dtype",
                        tag: t,
                    });
                }
            })
        }
        tags::CHAN_TAKE => Op::ChanTake(r.u32()?),
        tags::CHAN_READ => Op::ChanRead(r.u32()?),
        tags::CHAN_PUT => Op::ChanPut {
            chan: r.u32()?,
            value: r.u32()?,
        },
        tags::INTRINSIC_VAL => {
            let iv = r.u16()?;
            let intr = IntrinsicId::from_u16(iv).ok_or(ContainerDecodeError::UnknownTag {
                what: "intrinsic",
                tag: iv as u8,
            })?;
            let dtype = decode_dtype(r.u8()?)?;
            let shape = decode_shape(r)?;
            Op::IntrinsicVal { intr, shape, dtype }
        }
        tags::KERNEL_CALL => {
            let name = r.u16()?;
            let dtype = decode_dtype(r.u8()?)?;
            let shape = decode_shape(r)?;
            let n = r.u8()? as usize;
            let mut args = Vec::with_capacity(n);
            for _ in 0..n {
                args.push(r.u32()?);
            }
            Op::KernelCall {
                name,
                args,
                shape,
                dtype,
            }
        }
        tags::SINK_CALL => {
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
        t => {
            return Err(ContainerDecodeError::UnknownTag {
                what: "rng kind",
                tag: t,
            });
        }
    })
}

fn decode_dtype(t: u8) -> Result<DType, ContainerDecodeError> {
    Ok(match t {
        0 => DType::F32,
        1 => DType::I32,
        2 => DType::U32,
        3 => DType::Bool,
        t => {
            return Err(ContainerDecodeError::UnknownTag {
                what: "dtype",
                tag: t,
            });
        }
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
    Shape::new(&dims[..rank as usize]).ok_or(ContainerDecodeError::ZeroDimension)
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
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
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
        // `representatives()` is one op per `declare_ops!` row, so the wire
        // sweep cannot fall behind the table. The extra `Const` literals are
        // payload variation, not table coverage: `Literal` has four arms and
        // the row can only carry one.
        let mut ops = alloc::vec![
            Op::Const(Literal::I32(-1)),
            Op::Const(Literal::U32(7)),
            Op::Const(Literal::Bool(true)),
        ];
        ops.extend(crate::op::representatives());
        // `table_matches_op_metadata` pins the table's metadata; this pins
        // the *wire* path, and nothing else does. Without the sweep a new op
        // could land in `declare_ops!` with an `encode_op` arm and no
        // `decode_op` arm, and the first thing to read the missing half back
        // would be a driver.
        let missing: Vec<&str> = crate::op::OP_TABLE
            .iter()
            .filter(|spec| !ops.iter().any(|op| op.tag() == spec.tag))
            .map(|spec| spec.name)
            .collect();
        assert!(
            missing.is_empty(),
            "{} op(s) never round-trip through the container: {missing:?}",
            missing.len()
        );
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
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops,
            }],
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
        b[4] = 3;
        assert_eq!(decode(&b), Err(ContainerDecodeError::UnsupportedVersion(3)));
        let mut b = encode(&sample());
        b[4] = 9;
        assert_eq!(decode(&b), Err(ContainerDecodeError::UnsupportedVersion(9)));
        let b = encode(&sample());
        assert_eq!(
            decode(&b[..b.len() - 2]),
            Err(ContainerDecodeError::UnexpectedEof)
        );
    }

    #[test]
    fn retired_nucleus_opcode_is_unknown() {
        let mut reader = Reader::new(&[0x59]);
        assert_eq!(
            decode_op(&mut reader),
            Err(ContainerDecodeError::UnknownOpcode(0x59))
        );
    }

    /// The dual of `round_trip_every_op`: the 201 byte values `declare_ops!`
    /// does not allocate must all come back as `UnknownOpcode`, not as a
    /// neighbouring op that happens to share a decode arm. `0x59` above pins
    /// one of them by hand; a retired tag is only the case anyone thinks to
    /// write down.
    #[test]
    fn decode_rejects_every_tag_the_table_does_not_declare() {
        for tag in 0u8..=u8::MAX {
            if crate::op::OP_TABLE.iter().any(|spec| spec.tag == tag) {
                continue;
            }
            let bytes = [tag];
            let mut reader = Reader::new(&bytes);
            assert_eq!(
                decode_op(&mut reader),
                Err(ContainerDecodeError::UnknownOpcode(tag)),
                "tag {tag:#04x} is not in OP_TABLE but decode_op accepted it"
            );
        }
    }

    #[test]
    fn rejects_noncanonical_encodings() {
        assert!(Shape::new(&[0]).is_none());
        let minimal = TraceContainer {
            channels: vec![ChannelDecl {
                shape: Shape::SCALAR,
                dtype: ChanDType::Concrete(DType::U32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            }],
            ..TraceContainer::default()
        };

        let mut flags = minimal.encode();
        flags[6] = 1;
        assert_eq!(decode(&flags), Err(ContainerDecodeError::NonCanonical));

        let mut seeded = minimal.encode();
        seeded[31] = 2;
        assert_eq!(decode(&seeded), Err(ContainerDecodeError::NonCanonical));

        let mut empty_v2 = minimal.encode();
        empty_v2[4..6].copy_from_slice(&PTIR_VERSION_EXTERN.to_le_bytes());
        empty_v2.splice(24..24, 0u32.to_le_bytes());
        assert_eq!(decode(&empty_v2), Err(ContainerDecodeError::NonCanonical));
    }

    #[test]
    fn rejects_wire_counts_before_allocating_from_them() {
        let mut names = TraceContainer::default().encode();
        names[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            decode(&names),
            Err(ContainerDecodeError::CountTooLarge("name table"))
        );

        let mut operations = TraceContainer {
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: Vec::new(),
            }],
            ..TraceContainer::default()
        }
        .encode();
        operations[25..29].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            decode(&operations),
            Err(ContainerDecodeError::CountTooLarge("operation table"))
        );
    }
}
