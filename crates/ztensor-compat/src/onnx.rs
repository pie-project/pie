//! ONNX → zTensor object model projection.
//!
//! Extracts graph initializers (the model weights) from the protobuf
//! stream with a minimal hand-written wire-format parser, so no protobuf
//! dependency. A `raw_data` tensor is a plain range of the file, so it gets an
//! address and a borrow; the typed repeated fields (`float_data`,
//! `int32_data`, ...) have to be converted to little-endian bytes per the ONNX
//! storage rules (small types are stored one element per int32), so they exist
//! only once this reader has built them, so they are an opaque payload and
//! report as much.
//!
//! Graphs, nodes, and attributes are out of scope: this reads weights,
//! not computation. External data files are refused, not resolved.

use std::borrow::Cow;

use ztensor::provide::{Catalog, Decode};
use ztensor::provide::{Entry, Location, Payload};
use ztensor::{Error, Leaf, Result, Store, StoreId};

use crate::project::Projection;

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("onnx: {}", detail.into()))
}

/// ONNX TensorProto.DataType → leaf.
fn map_dtype(id: u64) -> Result<Leaf> {
    Ok(match id {
        1 => Leaf::F32,
        2 => Leaf::U8,
        3 => Leaf::I8,
        4 => Leaf::U16,
        5 => Leaf::I16,
        6 => Leaf::I32,
        7 => Leaf::I64,
        9 => Leaf::Bool,
        10 => Leaf::F16,
        11 => Leaf::F64,
        12 => Leaf::U32,
        13 => Leaf::U64,
        16 => Leaf::BF16,
        17 => Leaf::E4M3,
        19 => Leaf::E5M2,
        18 | 20 => {
            return Err(Error::Unsupported(
                "onnx fnuz fp8 (FLOAT8E4M3FNUZ/FLOAT8E5M2FNUZ) has no leaf; nothing \
                 in the registry denotes those bit patterns"
                    .into(),
            ))
        }
        other => {
            return Err(Error::Unsupported(format!(
                "onnx data type {other} has no registered projection"
            )))
        }
    })
}

// ---- protobuf wire format ---------------------------------------------

const VARINT: u32 = 0;
const I64BIT: u32 = 1;
const LEN: u32 = 2;
const I32BIT: u32 = 5;

struct Pb<'a> {
    data: &'a [u8],
    pos: usize,
    /// Absolute offset of `data[0]` in the file, for zero-copy ranges.
    base: usize,
}

impl<'a> Pb<'a> {
    fn done(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = *self
                .data
                .get(self.pos)
                .ok_or_else(|| bad("truncated varint"))?;
            self.pos += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift >= 64 {
                return Err(bad("varint too long"));
            }
        }
    }

    fn tag(&mut self) -> Result<(u32, u32)> {
        let v = self.varint()?;
        Ok(((v >> 3) as u32, (v & 7) as u32))
    }

    fn bytes(&mut self) -> Result<(usize, &'a [u8])> {
        let len = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| bad("length-delimited field truncated"))?;
        let abs = self.base + self.pos;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok((abs, s))
    }

    fn skip(&mut self, wire: u32) -> Result<()> {
        match wire {
            VARINT => {
                self.varint()?;
            }
            I64BIT => {
                self.pos = self
                    .pos
                    .checked_add(8)
                    .filter(|&e| e <= self.data.len())
                    .ok_or_else(|| bad("truncated fixed64"))?;
            }
            LEN => {
                self.bytes()?;
            }
            I32BIT => {
                self.pos = self
                    .pos
                    .checked_add(4)
                    .filter(|&e| e <= self.data.len())
                    .ok_or_else(|| bad("truncated fixed32"))?;
            }
            other => return Err(bad(format!("unknown wire type {other}"))),
        }
        Ok(())
    }
}

// ---- TensorProto ------------------------------------------------------

enum TensorData {
    Raw { offset: u64, length: u64 },
    Owned(Vec<u8>),
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    data_type: u64,
    data: TensorData,
}

fn parse_tensor(data: &[u8], base: usize) -> Result<TensorInfo> {
    let mut pb = Pb { data, pos: 0, base };
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    let mut raw: Option<(u64, u64)> = None;
    // The typed repeated fields, kept as their wire integers until the
    // data type is known.
    let mut i32s: Vec<u32> = Vec::new();
    let mut i64s: Vec<u64> = Vec::new();
    let mut f32s: Vec<u8> = Vec::new();
    let mut f64s: Vec<u8> = Vec::new();
    let mut external = false;

    while !pb.done() {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (1, LEN) => {
                let (_, b) = pb.bytes()?;
                let mut sub = Pb {
                    data: b,
                    pos: 0,
                    base: 0,
                };
                while !sub.done() {
                    dims.push(sub.varint()?);
                }
            }
            (1, VARINT) => dims.push(pb.varint()?),
            (2, VARINT) => data_type = pb.varint()?,
            (4, LEN) => f32s.extend_from_slice(pb.bytes()?.1), // packed floats: LE bytes
            (5, LEN) => {
                let (_, b) = pb.bytes()?;
                let mut sub = Pb {
                    data: b,
                    pos: 0,
                    base: 0,
                };
                while !sub.done() {
                    i32s.push(sub.varint()? as u32);
                }
            }
            (5, VARINT) => i32s.push(pb.varint()? as u32),
            (7, LEN) | (11, LEN) => {
                let (_, b) = pb.bytes()?;
                let mut sub = Pb {
                    data: b,
                    pos: 0,
                    base: 0,
                };
                while !sub.done() {
                    i64s.push(sub.varint()?);
                }
            }
            (7, VARINT) | (11, VARINT) => i64s.push(pb.varint()?),
            (8, LEN) => {
                name = String::from_utf8(pb.bytes()?.1.to_vec())
                    .map_err(|_| bad("tensor name is not UTF-8"))?;
            }
            (9, LEN) => {
                let (abs, b) = pb.bytes()?;
                raw = Some((abs as u64, b.len() as u64));
            }
            (10, LEN) => f64s.extend_from_slice(pb.bytes()?.1), // packed doubles
            (13, LEN) => {
                // external_data entries: presence alone means the payload
                // lives in another file.
                external = true;
                pb.bytes()?;
            }
            (14, VARINT) => {
                if pb.varint()? == 1 {
                    external = true;
                }
            }
            (_, w) => pb.skip(w)?,
        }
    }

    if external {
        return Err(Error::Unsupported(format!(
            "onnx: tensor {name:?} uses external data; internalize it first \
             (onnx.load_external_data_for_model)"
        )));
    }

    let width = map_dtype(data_type)?
        .width()
        .expect("onnx leaves are whole bytes") as usize;

    // Assemble owned bytes from typed fields per ONNX storage rules:
    // int32_data carries every type of width ≤ 4 (one element per entry),
    // int64_data carries i64, uint64_data carries u32/u64, float/double
    // are packed IEEE bytes.
    if raw.is_some() && !(f32s.is_empty() && f64s.is_empty() && i32s.is_empty() && i64s.is_empty())
    {
        return Err(bad(format!(
            "tensor {name:?} carries both raw_data and a typed data field"
        )));
    }
    let data = if let Some((offset, length)) = raw {
        TensorData::Raw { offset, length }
    } else if !f32s.is_empty() {
        TensorData::Owned(f32s)
    } else if !f64s.is_empty() {
        TensorData::Owned(f64s)
    } else if !i32s.is_empty() {
        let mut out = Vec::with_capacity(i32s.len() * width);
        for v in &i32s {
            out.extend_from_slice(&v.to_le_bytes()[..width.min(4)]);
        }
        TensorData::Owned(out)
    } else if !i64s.is_empty() {
        let mut out = Vec::with_capacity(i64s.len() * width);
        for v in &i64s {
            out.extend_from_slice(&v.to_le_bytes()[..width.min(8)]);
        }
        TensorData::Owned(out)
    } else {
        TensorData::Owned(Vec::new())
    };

    Ok(TensorInfo {
        name,
        dims,
        data_type,
        data,
    })
}

// ---- projection -------------------------------------------------------

/// The typed repeated fields, converted once at open and handed out on
/// request. They have no address: nothing in the file holds these bytes in the
/// layout a consumer wants.
struct Typed {
    buffers: Vec<Vec<u8>>,
}

impl Decode for Typed {
    fn decode(&self, key: u64, decoded_len: u64) -> Result<Vec<u8>> {
        let bytes = self
            .buffers
            .get(key as usize)
            .ok_or_else(|| bad(format!("no converted tensor {key}")))?;
        if bytes.len() as u64 != decoded_len {
            return Err(bad("converted tensor changed size"));
        }
        Ok(bytes.clone())
    }
}

pub(crate) fn project(store: &Store) -> Result<Projection> {
    // ONNX keeps its weights inline in the protobuf stream and gives no index,
    // so there is no header to read on its own: an unmapped store has to read
    // the file to answer anything at all.
    let bytes: Cow<'_, [u8]> = match store.bytes() {
        Some(mapped) => Cow::Borrowed(mapped),
        None => Cow::Owned(store.read(0, store.len())?),
    };
    let buf: &[u8] = &bytes;
    if buf.is_empty() {
        return Err(bad("empty file"));
    }
    // ModelProto.graph is field 7.
    let mut pb = Pb {
        data: buf,
        pos: 0,
        base: 0,
    };
    let mut graph: Option<(usize, usize)> = None;
    while !pb.done() {
        let (field, wire) = pb.tag()?;
        if field == 7 && wire == LEN {
            let (abs, b) = pb.bytes()?;
            graph = Some((abs, b.len()));
            break;
        }
        pb.skip(wire)?;
    }
    let (graph_base, graph_len) = graph.ok_or_else(|| bad("no graph field in ModelProto"))?;
    let graph_bytes = &buf[graph_base..graph_base + graph_len];

    // GraphProto.initializer is field 5.
    let mut catalog = Catalog::new();
    let mut converted: Vec<Vec<u8>> = Vec::new();
    let mut pb = Pb {
        data: graph_bytes,
        pos: 0,
        base: graph_base,
    };
    while !pb.done() {
        let (field, wire) = pb.tag()?;
        if field != 5 || wire != LEN {
            pb.skip(wire)?;
            continue;
        }
        let (abs, b) = pb.bytes()?;
        let info = parse_tensor(b, abs)?;
        if info.name.is_empty() {
            continue;
        }
        let leaf = map_dtype(info.data_type)?;
        let elems = crate::safe::product("onnx shape", &info.dims)?;
        let expected = leaf
            .size(elems)
            .ok_or_else(|| bad("size not computable"))?;
        let actual = match &info.data {
            TensorData::Raw { length, .. } => *length,
            TensorData::Owned(v) => v.len() as u64,
        };
        if actual != expected {
            return Err(bad(format!(
                "tensor {:?} holds {actual} bytes but shape implies {expected}",
                info.name
            )));
        }

        let payload = match info.data {
            TensorData::Raw { offset, length } => Payload::At(Location {
                store: StoreId(0),
                offset,
                len: length,
            }),
            TensorData::Owned(v) => {
                converted.push(v);
                Payload::Opaque {
                    store: StoreId(0),
                    key: converted.len() as u64 - 1,
                    decoded_len: expected,
                }
            }
        };

        if catalog
            .insert(
                info.name.clone(),
                Entry {
                    shape: info.dims,
                    term: Some(leaf.into()),
                    layout: None,
                    attributes: None,
                    payload,
                    digest: None,
                    blocks: None,
                },
            )
            .is_some()
        {
            return Err(bad(format!("duplicate initializer {:?}", info.name)));
        }
    }

    // Weights are embedded in a protobuf stream with fields around and between
    // them, so what else shares a page is not knowable from here: occupancy
    // stays unstated and exclusivity is never claimed.
    let projection = Projection::new(catalog);
    Ok(if converted.is_empty() {
        projection
    } else {
        projection.with_decoder(Box::new(Typed { buffers: converted }))
    })
}
