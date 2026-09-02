//! safetensors → the zTensor object model.
//!
//! The format: `[header_len: u64 LE][JSON header][data]`, where the header
//! maps tensor names to `{dtype, shape, data_offsets: [begin, end]}` with
//! offsets relative to the data section.
//!
//! This is a deliberately strict reader. safetensors headers are JSON, and
//! JSON parsers resolve duplicate keys silently (the classic safetensors
//! aliasing attack); we defuse that class entirely by requiring the tensor
//! ranges to tile the data section exactly: sorted, gap-free, overlap-free,
//! ending at EOF. A file that fails any of it is rejected.
//!
//! Every tensor is a raw range of the file, so every one of them gets a
//! [`Payload::At`]: addressable, mappable, and evictable where the ranges
//! happen to land on pages. What they never get is a digest, because the
//! format carries none.

use serde_json::Value as Json;
use ztensor::format::cbor::Value;
use ztensor::provide::Catalog;
use ztensor::provide::{Entry, Location};
use ztensor::{Error, Leaf, Result, Store, StoreId};

use crate::project::Projection;
use crate::safe;

/// Practical cap on the JSON header (matches the reference implementation).
const MAX_HEADER: u64 = 100 << 20;

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("safetensors: {}", detail.into()))
}

/// safetensors dtype → leaf (spec Appendix A).
fn map_dtype(st: &str) -> Result<Leaf> {
    Ok(match st {
        "F64" => Leaf::F64,
        "F32" => Leaf::F32,
        "F16" => Leaf::F16,
        "BF16" => Leaf::BF16,
        "I64" => Leaf::I64,
        "I32" => Leaf::I32,
        "I16" => Leaf::I16,
        "I8" => Leaf::I8,
        "U64" => Leaf::U64,
        "U32" => Leaf::U32,
        "U16" => Leaf::U16,
        "U8" => Leaf::U8,
        "BOOL" => Leaf::Bool,
        "F8_E4M3" => Leaf::E4M3,
        "F8_E5M2" => Leaf::E5M2,
        // Packed two per byte, lsb-first, in both formats.
        "F4_E2M1" => Leaf::E2M1,
        "F8_E8M0" => Leaf::E8M0,
        other => {
            return Err(Error::Unsupported(format!(
                "safetensors dtype {other:?} has no registered projection"
            )))
        }
    })
}

pub(crate) fn project(store: &Store) -> Result<Projection> {
    let file_len = store.len();
    if file_len < 8 {
        return Err(bad("file shorter than the header length field"));
    }
    let header_len = u64::from_le_bytes(store.read(0, 8)?.try_into().unwrap());
    if header_len > MAX_HEADER {
        return Err(bad(format!("header length {header_len} exceeds cap")));
    }
    let data_start = safe::add("safetensors header", 8, header_len)?;
    if data_start > file_len {
        return Err(bad("header extends past end of file"));
    }
    let data_len = file_len - data_start;

    let header: Json = serde_json::from_slice(&store.read(8, header_len)?)
        .map_err(|e| bad(format!("header is not valid JSON: {e}")))?;
    let Json::Object(entries) = header else {
        return Err(bad("header root must be a JSON object"));
    };

    let mut attributes: Vec<(Value, Value)> = Vec::new();
    let mut catalog = Catalog::new();
    // (begin, end) for the exact-tiling check, in data-section coordinates.
    let mut ranges: Vec<(u64, u64)> = Vec::new();

    for (name, entry) in entries {
        if name == "__metadata__" {
            // Some writers (MLX conversions among them) emit `"__metadata__":
            // null` for "no metadata"; that reads as an empty block.
            if entry.is_null() {
                continue;
            }
            let Json::Object(meta) = entry else {
                return Err(bad("__metadata__ must be an object"));
            };
            for (k, v) in meta {
                let Json::String(s) = v else {
                    return Err(bad("__metadata__ values must be strings"));
                };
                attributes.push((Value::Text(k), Value::Text(s)));
            }
            continue;
        }

        let Json::Object(fields) = entry else {
            return Err(bad(format!("tensor {name:?} must be an object")));
        };
        let dtype_str = fields
            .get("dtype")
            .and_then(Json::as_str)
            .ok_or_else(|| bad(format!("tensor {name:?} missing dtype")))?;
        let leaf = map_dtype(dtype_str)?;
        let shape: Vec<u64> = fields
            .get("shape")
            .and_then(Json::as_array)
            .ok_or_else(|| bad(format!("tensor {name:?} missing shape")))?
            .iter()
            .map(|d| d.as_u64())
            .collect::<Option<_>>()
            .ok_or_else(|| bad(format!("tensor {name:?} has non-integer dims")))?;
        let offsets = fields
            .get("data_offsets")
            .and_then(Json::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| bad(format!("tensor {name:?} missing data_offsets")))?;
        let (begin, end) = match (offsets[0].as_u64(), offsets[1].as_u64()) {
            (Some(b), Some(e)) if b <= e && e <= data_len => (b, e),
            _ => return Err(bad(format!("tensor {name:?} has invalid data_offsets"))),
        };

        let elems = safe::product("safetensors shape", &shape)?;
        let expected = leaf
            .size(elems)
            .ok_or_else(|| bad(format!("tensor {name:?} size not computable")))?;
        if end - begin != expected {
            return Err(bad(format!(
                "tensor {name:?} holds {} bytes but shape implies {expected}",
                end - begin
            )));
        }

        ranges.push((begin, end));
        let at = Location {
            store: StoreId(0),
            offset: data_start + begin, // absolute file offset
            len: end - begin,
        };
        if catalog
            .insert(name.clone(), Entry::leaf(shape, leaf, at))
            .is_some()
        {
            return Err(bad(format!("duplicate tensor name {name:?}")));
        }
    }

    // Exact tiling of the data section: sorted, gap-free, overlap-free, ending
    // at EOF. This forecloses aliasing regardless of how the JSON parser
    // resolved duplicate keys.
    ranges.sort_unstable();
    let mut cursor = 0u64;
    for (begin, end) in &ranges {
        if *begin != cursor {
            return Err(bad(format!(
                "data section not tiled exactly: range starts at {begin}, expected {cursor}"
            )));
        }
        cursor = *end;
    }
    if cursor != data_len {
        return Err(bad(format!(
            "data section is {data_len} bytes but tensors cover {cursor}"
        )));
    }

    if !attributes.is_empty() {
        catalog.set_attributes(Some(Value::Map(attributes)));
    }

    // The header occupies everything before the first tensor, and the tensors
    // tile the rest, so the occupancy map is exact and page exclusivity is a
    // fact about this file rather than an assumption.
    let mut occupied = vec![(0, data_start)];
    occupied.extend(
        ranges
            .iter()
            .filter(|(b, e)| e > b)
            .map(|(b, e)| (data_start + b, e - b)),
    );
    Ok(Projection::new(catalog).occupying(occupied))
}
