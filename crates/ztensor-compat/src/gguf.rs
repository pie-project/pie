//! GGUF → zTensor object model projection.
//!
//! Layout: `"GGUF"` magic, u32 version (2 or 3, little-endian), u64 tensor
//! count, u64 metadata KV count, the KVs, the tensor infos, then the data
//! section aligned to `general.alignment` (default 32).
//!
//! Projection choices:
//! - Standard element types project to the leaf in canonical layout.
//! - Quantized tensors keep their **logical shape** and get layout
//!   `gguf.<type>/2`, the type the block format expresses (none for the
//!   codebook types), the raw blocks verbatim as the blob, and the
//!   `elems_per_block` / `block_bytes` attributes the layout requires. Nothing
//!   is dequantized; unknown type ids reject the file (never reinterpret).
//! - All metadata KVs (including tokenizer tables) become file attributes.

use ztensor::format::cbor::Value;
use ztensor::provide::Catalog;
use ztensor::provide::{Entry, Location, Payload};
use ztensor::vocab::gguf::{row_of, Row};
use ztensor::{Error, Leaf, Result, Store, StoreId};

use crate::project::Projection;

/// The cursor ran past the window we had read, not past the file. Only
/// meaningful while the window is smaller than the file.
const NEED_MORE: &str = "header extends past the bytes read so far";

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("gguf: {}", detail.into()))
}

/// ggml type id → ggml type name (ggml.h `ggml_type`). Block geometry comes
/// from the core's `gguf.<type>/2` table, so only the numbering lives here.
fn type_name(id: u32) -> Result<&'static str> {
    Ok(match id {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        15 => "q8_k",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        39 => "mxfp4",
        other => {
            return Err(Error::Unsupported(format!(
                "gguf tensor type id {other} has no registered projection"
            )))
        }
    })
}

/// What a ggml type projects to: a leaf, or a `gguf.<type>/2` layout row.
enum Kind {
    Leaf(Leaf),
    Block(&'static Row),
}

impl Kind {
    fn of(id: u32) -> Result<Kind> {
        let name = type_name(id)?;
        if let Some(leaf) = Leaf::parse(name) {
            return Ok(Kind::Leaf(leaf));
        }
        row_of(name).map(Kind::Block).ok_or_else(|| {
            Error::Unsupported(format!("gguf type {name:?} has no gguf.{name}/2 layout"))
        })
    }

    fn geometry(&self) -> (u64, u64) {
        match self {
            Kind::Leaf(leaf) => (1, leaf.width().expect("ggml element types are whole bytes")),
            Kind::Block(row) => (row.elems_per_block, row.block_bytes),
        }
    }
}

// ---- cursor -----------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| bad(NEED_MORE))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > self.data.len() as u64 {
            return Err(bad("string length exceeds file"));
        }
        String::from_utf8(self.take(len as usize)?.to_vec())
            .map_err(|_| bad("invalid UTF-8 in string"))
    }

    /// Parses one metadata value into a CBOR attribute value.
    fn meta_value(&mut self, vtype: u32, depth: u32) -> Result<Value> {
        if depth > 32 {
            return Err(bad("metadata nesting too deep"));
        }
        Ok(match vtype {
            0 => Value::Uint(self.take(1)?[0] as u64),
            1 => int_value(self.take(1)?[0] as i8 as i64),
            2 => Value::Uint(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => int_value(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Uint(self.u32()? as u64),
            5 => int_value(i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Text(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()?;
                // Even a 1-byte element type needs a byte on disk, so a
                // count beyond the remaining bytes is a lie, and the
                // materialized `Value`s are far larger than their encoding,
                // so this bound is what keeps the projection proportional
                // to the file.
                let remaining = (self.data.len() - self.pos) as u64;
                if count > remaining {
                    return Err(bad("array length exceeds remaining bytes"));
                }
                let mut items = Vec::with_capacity(count.min(1 << 16) as usize);
                for _ in 0..count {
                    items.push(self.meta_value(elem_type, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::Uint(self.u64()?),
            11 => int_value(self.u64()? as i64),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(bad(format!("unknown metadata value type {other}"))),
        })
    }
}

fn int_value(v: i64) -> Value {
    if v < 0 {
        Value::Nint((-1 - v) as u64)
    } else {
        Value::Uint(v as u64)
    }
}

// ---- projection -------------------------------------------------------

/// Reads the header, growing the window until it fits.
///
/// A GGUF header is metadata KVs plus tensor infos, and its size is only known
/// once it has been parsed, and a tokenizer table can be megabytes. So the window
/// doubles until the parse stops running off the end, which for a mapped file
/// costs nothing and for an indexed one reads the header and not the 100 GB
/// behind it.
pub(crate) fn project(store: &Store) -> Result<Projection> {
    let file_len = store.len();
    if let Some(mapped) = store.bytes() {
        return project_bytes(mapped, file_len);
    }
    let mut window = (1u64 << 20).min(file_len);
    loop {
        let buf = store.read(0, window)?;
        match project_bytes(&buf, file_len) {
            Err(e) if window < file_len && format!("{e}").contains(NEED_MORE) => {
                window = (window * 4).min(file_len);
            }
            other => return other,
        }
    }
}

/// `buf` is the file, or a prefix of it long enough to hold the header;
/// `file_len` is always the whole file, since tensor bounds are checked
/// against the file and not against what we happened to read.
fn project_bytes(buf: &[u8], file_len: u64) -> Result<Projection> {
    let mut c = Cursor { data: buf, pos: 0 };
    if c.take(4)? != b"GGUF" {
        return Err(bad("bad magic"));
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        return Err(Error::Unsupported(format!(
            "gguf version {version} (little-endian v2/v3 only)"
        )));
    }
    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;
    if tensor_count > file_len || kv_count > file_len {
        return Err(bad("header counts exceed file size"));
    }

    let mut alignment = 32u64;
    // A KV needs at least a 8-byte length + 4-byte type on disk; a count
    // beyond that is a lie and must not drive the allocation.
    let mut attributes: Vec<(Value, Value)> =
        Vec::with_capacity(crate::safe::capacity(kv_count, 13, buf.len()));
    for _ in 0..kv_count {
        let key = c.string()?;
        let vtype = c.u32()?;
        let value = c.meta_value(vtype, 0)?;
        if key == "general.alignment" {
            alignment = value
                .as_u64()
                .filter(|a| a.is_power_of_two())
                .ok_or_else(|| bad("general.alignment must be a power-of-two uint"))?;
        }
        attributes.push((Value::Text(key), value));
    }

    struct Info {
        name: String,
        shape: Vec<u64>,
        type_id: u32,
        offset: u64,
    }
    let mut infos = Vec::with_capacity(crate::safe::capacity(tensor_count, 24, buf.len()));
    for _ in 0..tensor_count {
        let name = c.string()?;
        let n_dims = c.u32()?;
        if n_dims > 64 {
            return Err(bad(format!("tensor {name:?} has {n_dims} dims")));
        }
        // ggml stores dims fastest-first; reverse to row-major.
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(c.u64()?);
        }
        shape.reverse();
        let type_id = c.u32()?;
        let offset = c.u64()?;
        infos.push(Info {
            name,
            shape,
            type_id,
            offset,
        });
    }

    let data_start = crate::safe::mul(
        "gguf data section",
        (c.pos as u64).div_ceil(alignment),
        alignment,
    )?;
    if data_start > file_len {
        return Err(bad("aligned data section starts past end of file"));
    }

    let mut catalog = Catalog::new();
    let mut ranges = vec![(0u64, data_start)]; // header region
    for info in infos {
        let kind = Kind::of(info.type_id)?;
        let (epb, block_bytes) = kind.geometry();
        // Blocks are per-row in ggml: the fastest dim must divide evenly.
        let fastest = info.shape.last().copied().unwrap_or(1);
        if fastest % epb != 0 {
            return Err(bad(format!(
                "tensor {:?}: fastest dim {fastest} not divisible by block size {epb}",
                info.name
            )));
        }
        let elems = crate::safe::product("gguf shape", &info.shape)?;
        let byte_size = crate::safe::mul("gguf tensor size", elems / epb, block_bytes)?;
        let abs = crate::safe::add("gguf tensor offset", data_start, info.offset)?;
        if crate::safe::add("gguf tensor", abs, byte_size)? > file_len {
            return Err(bad(format!(
                "tensor {:?} extends past end of file",
                info.name
            )));
        }

        let (term, layout, attributes) = match kind {
            Kind::Leaf(leaf) => (Some(leaf.into()), None, None),
            Kind::Block(row) => (
                row.term(),
                Some(row.layout_id()),
                Some(Value::Map(vec![
                    (Value::Text("elems_per_block".into()), Value::Uint(epb)),
                    (Value::Text("block_bytes".into()), Value::Uint(block_bytes)),
                ])),
            ),
        };
        ranges.push((abs, byte_size));
        if catalog
            .insert(
                info.name.clone(),
                Entry {
                    shape: info.shape,
                    term,
                    layout,
                    attributes,
                    payload: Payload::At(Location {
                        store: StoreId(0),
                        offset: abs,
                        len: byte_size,
                    }),
                    digest: None, // gguf carries none
                    blocks: None,
                },
            )
            .is_some()
        {
            return Err(bad(format!("duplicate tensor name {:?}", info.name)));
        }
    }

    // Ranges must not overlap (offsets are writer-controlled).
    ranges.sort_unstable();
    ranges.dedup();
    for w in ranges.windows(2) {
        if w[0].0 + w[0].1 > w[1].0 {
            return Err(bad(format!(
                "tensor ranges overlap at {} and {}",
                w[0].0, w[1].0
            )));
        }
    }

    if !attributes.is_empty() {
        catalog.set_attributes(Some(Value::Map(attributes)));
    }
    Ok(Projection::new(catalog).occupying(ranges))
}
