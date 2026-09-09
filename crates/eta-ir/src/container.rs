use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::op::{self, ChannelIndex, IntrinsicId, Op, WireField};
use super::read::{ReadError, Reader};
use super::registry::{Port, Stage};
use super::wire::{OpWire, predicate_tags};
use crate::types::{Dtype, MAX_RANK, RngKind, Shape, from_wire, to_wire, wire_dtype};
use crate::{ETA_MAGIC, ETA_VERSION, ETA_VERSION_EXTERN};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChanDType {
    Concrete(Dtype),
    Act,
}

pub const DT_ACT: u8 = 4;

impl ChanDType {
    pub fn tag(self) -> Option<u8> {
        match self {
            ChanDType::Concrete(d) => to_wire(d),
            ChanDType::Act => Some(DT_ACT),
        }
    }
    pub fn from_tag(t: u8) -> Option<Self> {
        if t == DT_ACT {
            return Some(ChanDType::Act);
        }
        from_wire(t).map(ChanDType::Concrete)
    }
    pub fn program_dtype(self) -> Dtype {
        match self {
            ChanDType::Concrete(d) => d,
            ChanDType::Act => Dtype::F32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum HostRole {
    None = 0,
    Writer = 1,
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

#[derive(Clone, Debug, PartialEq)]
pub struct ChannelDecl {
    pub shape: Shape,
    pub dtype: ChanDType,
    pub capacity: u32,
    pub host_role: HostRole,
    pub seeded: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PortSource {
    Channel(ChannelIndex),
    Const {
        dtype: Dtype,
        shape: Shape,
        data: Vec<u8>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PortBinding {
    pub port: Port,
    pub source: PortSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum ExternDir {
    Import = 0,
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

#[derive(Clone, Debug, PartialEq)]
pub struct ExternDecl {
    pub name: crate::op::NameIndex,
    pub dir: ExternDir,
    pub chan: ChannelIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageProgram {
    pub stage: Stage,
    pub ops: Vec<Op>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct TraceContainer {
    pub names: Vec<String>,
    pub channels: Vec<ChannelDecl>,
    pub ports: Vec<PortBinding>,
    pub stages: Vec<StageProgram>,
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

fn wire_len<T>(len: usize, table: &str) -> T
where
    T: TryFrom<usize>,
{
    match T::try_from(len) {
        Ok(value) => value,
        Err(_) => panic!("{table} table of {len} entries exceeds its wire width"),
    }
}

pub fn encode(c: &TraceContainer) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(&ETA_MAGIC);
    let v2 = !c.externs.is_empty();
    put_u16(&mut w, if v2 { ETA_VERSION_EXTERN } else { ETA_VERSION });
    put_u16(&mut w, 0);
    put_u32(&mut w, wire_len(c.names.len(), "name"));
    put_u32(&mut w, wire_len(c.channels.len(), "channel"));
    put_u32(&mut w, wire_len(c.ports.len(), "port"));
    put_u32(&mut w, wire_len(c.stages.len(), "stage"));
    if v2 {
        put_u32(&mut w, wire_len(c.externs.len(), "extern"));
    }
    for n in &c.names {
        put_u16(&mut w, wire_len(n.len(), "name byte"));
        w.extend_from_slice(n.as_bytes());
    }
    for ch in &c.channels {
        w.push(match ch.dtype.tag() {
            Some(tag) => tag,
            None => wire_dtype(ch.dtype.program_dtype()),
        });
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
                w.push(wire_dtype(*dtype));
                encode_shape(&mut w, *shape);
                w.extend_from_slice(data);
            }
        }
    }
    for s in &c.stages {
        w.push(s.stage as u8);
        put_u32(&mut w, wire_len(s.ops.len(), "op"));
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
    let wire = OpWire::of(op);
    w.push(wire.tag);
    let layout = op::spec(wire.tag).expect("op tag has no OP_TABLE row").wire;
    let mut value = 0usize;
    let mut imm = 0usize;
    for field in layout {
        match field {
            WireField::Value => {
                put_u32(w, wire.args[value]);
                value += 1;
            }
            WireField::Chan => put_u32(
                w,
                u32::try_from(wire.chan).expect("a chan-carrying op records its channel index"),
            ),
            WireField::Imm => {
                put_u32(w, [wire.imm, wire.imm2, wire.imm3][imm]);
                imm += 1;
            }
            WireField::Dtype => w.push(wire.dtype),
            WireField::Shape => {
                w.push(wire_len(wire.shape.len(), "shape dim"));
                for &dim in &wire.shape {
                    put_u32(w, dim);
                }
            }
            WireField::RngKind => w.push(wire.kind),
            WireField::Predicate => {
                w.push(wire.pred_tag);
                put_u32(w, wire.pred_payload);
            }
            WireField::Literal => {
                w.push(wire.lit_dtype);
                put_u32(w, wire.lit_bits);
            }
            WireField::Name => put_u16(w, wire.name_idx),
            WireField::Intrinsic => put_u16(w, wire.intr),
            WireField::Args => {
                let rest = &wire.args[value..];
                w.push(wire_len(rest.len(), "operand"));
                for &arg in rest {
                    put_u32(w, arg);
                }
            }
        }
    }
}

pub fn encode_shape(w: &mut Vec<u8>, shape: Shape) {
    w.push(wire_len(shape.rank(), "shape dim"));
    for &d in shape.dims() {
        put_u32(w, d);
    }
}

pub fn put_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub fn const_elem_size(dtype: Dtype) -> usize {
    usize::try_from(dtype.bytes_ceil())
        .expect("an element's byte count fits a usize on every served target")
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
            BadMagic => f.write_str("bad magic (expected \"ETA\")"),
            UnsupportedVersion(v) => write!(f, "unsupported container version {v}"),
            UnexpectedEof => f.write_str("unexpected end of buffer"),
            UnknownOpcode(t) => write!(f, "unknown opcode 0x{t:02x}"),
            UnknownTag { what, tag } => write!(f, "unknown {what} tag 0x{tag:02x}"),
            RankTooLarge(r) => write!(f, "shape rank {r} exceeds MAX_RANK"),
            ZeroDimension => f.write_str("shape dimensions must be nonzero"),
            BadUtf8 => f.write_str("name table entry is not valid UTF-8"),
            TrailingBytes => f.write_str("trailing bytes after ETA container"),
            NonCanonical => f.write_str("noncanonical ETA container encoding"),
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

pub const MAX_STAGES: usize = Stage::ALL.len();
pub const MAX_OPS: usize = 1 << 16;
pub const MAX_CHANNELS: usize = 1 << 12;
pub const MAX_NAMES: usize = 1 << 12;
pub const MAX_PORTS: usize = 1 << 8;
pub const MAX_EXTERNS: usize = MAX_CHANNELS;

pub fn decode(bytes: &[u8]) -> Result<TraceContainer, ContainerDecodeError> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != ETA_MAGIC {
        return Err(ContainerDecodeError::BadMagic);
    }
    let version = r.u16()?;
    if version != ETA_VERSION && version != ETA_VERSION_EXTERN {
        return Err(ContainerDecodeError::UnsupportedVersion(version));
    }
    let _flags = r.u16()?;
    let n_names = r.u32()?;
    let n_channels = r.u32()?;
    let n_ports = r.u32()?;
    let n_stages = r.u32()?;
    let n_externs = if version == ETA_VERSION_EXTERN {
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
    let layout = op::spec(tag)
        .ok_or(ContainerDecodeError::UnknownOpcode(tag))?
        .wire;
    let mut wire = OpWire {
        tag,
        chan: -1,
        ..OpWire::default()
    };
    let mut imm = 0usize;
    for field in layout {
        match field {
            WireField::Value => wire.args.push(r.u32()?),
            WireField::Chan => wire.chan = i64::from(r.u32()?),
            WireField::Imm => {
                let value = r.u32()?;
                match imm {
                    0 => wire.imm = value,
                    1 => wire.imm2 = value,
                    _ => wire.imm3 = value,
                }
                imm += 1;
            }
            WireField::Dtype => {
                let byte = r.u8()?;
                decode_dtype(byte)?;
                wire.dtype = byte;
            }
            WireField::Shape => wire.shape = decode_shape(r)?.dims().to_vec(),
            WireField::RngKind => wire.kind = decode_rng_kind(r.u8()?)? as u8,
            WireField::Predicate => {
                let pred = r.u8()?;
                if pred > predicate_tags::PROB_GE {
                    return Err(ContainerDecodeError::UnknownTag {
                        what: "predicate",
                        tag: pred,
                    });
                }
                wire.pred_tag = pred;
                wire.pred_payload = r.u32()?;
            }
            WireField::Literal => {
                let dtype = r.u8()?;
                wire.lit_bits = r.u32()?;
                if from_wire(dtype).is_none() {
                    return Err(ContainerDecodeError::UnknownTag {
                        what: "literal dtype",
                        tag: dtype,
                    });
                }
                wire.lit_dtype = dtype;
            }
            WireField::Name => wire.name_idx = r.u16()?,
            WireField::Intrinsic => {
                let intr = r.u16()?;
                if IntrinsicId::from_u16(intr).is_none() {
                    return Err(ContainerDecodeError::UnknownTag {
                        what: "intrinsic (low byte)",
                        tag: intr.to_le_bytes()[0],
                    });
                }
                wire.intr = intr;
            }
            WireField::Args => {
                let count = r.u8()? as usize;
                wire.args.reserve(count);
                for _ in 0..count {
                    wire.args.push(r.u32()?);
                }
            }
        }
    }
    wire.to_op().ok_or(ContainerDecodeError::UnknownOpcode(tag))
}

fn decode_rng_kind(t: u8) -> Result<RngKind, ContainerDecodeError> {
    Ok(match t {
        0 => RngKind::Uniform,
        1 => RngKind::Gumbel,
        2 => RngKind::Normal,
        t => {
            return Err(ContainerDecodeError::UnknownTag {
                what: "rng kind",
                tag: t,
            });
        }
    })
}

fn decode_dtype(t: u8) -> Result<Dtype, ContainerDecodeError> {
    match from_wire(t) {
        Some(d) => Ok(d),
        None => Err(ContainerDecodeError::UnknownTag {
            what: "dtype",
            tag: t,
        }),
    }
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
    use crate::types::Literal;
    use alloc::string::ToString;
    use alloc::vec;

    fn sample() -> TraceContainer {
        let vocab = 32u32;
        TraceContainer {
            names: vec!["envelope_dot".to_string()],
            channels: vec![
                ChannelDecl {
                    shape: Shape::vector(1),
                    dtype: ChanDType::Concrete(Dtype::I32),
                    capacity: 1,
                    host_role: HostRole::None,
                    seeded: true,
                },
                ChannelDecl {
                    shape: Shape::vector(1),
                    dtype: ChanDType::Concrete(Dtype::I32),
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
                        dtype: Dtype::U32,
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
                        dtype: Dtype::F32,
                    },
                    Op::ReduceArgmax(0),
                    Op::ChanPut { chan: 0, value: 1 },
                    Op::ChanPut { chan: 1, value: 1 },
                ],
            }],
            externs: Vec::new(),
        }
    }

    #[test]
    fn container_every_case() {
        round_trip_every_op();
        no_byte_of_an_op_encoding_is_ignored_by_its_decoder();
        rejects_bad_magic_version_and_truncation();
        rejects_noncanonical_encodings();
        rejects_wire_counts_before_allocating_from_them();
    }

    fn round_trip_every_op() {
        let mut ops = alloc::vec![
            Op::Const(Literal::I32(-1)),
            Op::Const(Literal::U32(7)),
            Op::Const(Literal::Bool(true)),
        ];
        ops.extend(crate::op::representatives());
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
                    dtype: ChanDType::Concrete(Dtype::F32),
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

    fn no_byte_of_an_op_encoding_is_ignored_by_its_decoder() {
        let mut ignored: Vec<String> = Vec::new();
        let mut flipped = 0usize;
        for op in crate::op::representatives() {
            let mut bytes = alloc::vec::Vec::new();
            encode_op(&mut bytes, &op);
            for index in 1..bytes.len() {
                for mask in [0x01u8, 0x80, 0xff] {
                    let mut mutant = bytes.clone();
                    mutant[index] ^= mask;
                    flipped += 1;
                    let mut r = Reader::new(&mutant);
                    if let Ok(decoded) = decode_op(&mut r)
                        && decoded == op
                        && r.remaining() == 0
                    {
                        ignored.push(format!(
                            "{}: byte {index} ^ {mask:#04x} decodes back to \
                             the same op",
                            op.tag()
                        ));
                    }
                }
            }
        }
        assert!(
            flipped > 500,
            "only {flipped} flips; the representatives stopped carrying payload"
        );
        assert!(
            ignored.is_empty(),
            "{} encoded byte(s) do not reach the op they encode: {ignored:?}",
            ignored.len()
        );
    }

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

    fn rejects_noncanonical_encodings() {
        assert!(Shape::new(&[0]).is_none());
        let minimal = TraceContainer {
            channels: vec![ChannelDecl {
                shape: Shape::SCALAR,
                dtype: ChanDType::Concrete(Dtype::U32),
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
        empty_v2[4..6].copy_from_slice(&ETA_VERSION_EXTERN.to_le_bytes());
        empty_v2.splice(24..24, 0u32.to_le_bytes());
        assert_eq!(decode(&empty_v2), Err(ContainerDecodeError::NonCanonical));
    }

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
