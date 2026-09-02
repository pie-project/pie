//! NumPy `.npz` (and single `.npy`) → zTensor object model projection.
//!
//! An `.npz` is a ZIP of `.npy` entries. A stored (uncompressed) entry is a
//! plain range of the file, so it gets an address and a borrow like any other.
//! A deflated entry has neither, since its bytes do not exist until something
//! inflates them. It becomes an opaque payload: readable, and honest that
//! reading costs a decompression.
//!
//! Refusals (never reinterpret): big-endian descrs, `fortran_order: True`
//! (reversing the shape would silently transpose the data), object dtypes.

use std::fs::File;
use std::io::Read;

use ztensor::provide::{Catalog, Decode};
use ztensor::provide::{Entry, Location, Payload};
use ztensor::{Error, Leaf, Result, Store, StoreId};

use crate::project::Projection;

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("npz: {}", detail.into()))
}

/// numpy descr → leaf. Little-endian or byte-order-free only.
fn map_descr(descr: &str) -> Result<Leaf> {
    Ok(match descr {
        "<f8" | "=f8" => Leaf::F64,
        "<f4" | "=f4" => Leaf::F32,
        "<f2" | "=f2" => Leaf::F16,
        "<i8" | "=i8" => Leaf::I64,
        "<i4" | "=i4" => Leaf::I32,
        "<i2" | "=i2" => Leaf::I16,
        "|i1" => Leaf::I8,
        "<u8" | "=u8" => Leaf::U64,
        "<u4" | "=u4" => Leaf::U32,
        "<u2" | "=u2" => Leaf::U16,
        "|u1" => Leaf::U8,
        "|b1" => Leaf::Bool,
        other => {
            return Err(Error::Unsupported(format!(
                "numpy descr {other:?} has no projection (big-endian and object dtypes are refused)"
            )))
        }
    })
}

pub(crate) struct NpyHeader {
    pub leaf: Leaf,
    pub shape: Vec<u64>,
    /// Offset of raw data relative to the start of the `.npy` bytes.
    pub data_offset: usize,
}

/// Parses a `.npy` header (magic, version, header dict).
pub(crate) fn parse_npy_header(data: &[u8]) -> Result<NpyHeader> {
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy entry"));
    }
    let major = data[6];
    let (header_len, header_start): (usize, usize) = if major >= 2 {
        if data.len() < 12 {
            return Err(bad("truncated header"));
        }
        (
            u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize,
            12,
        )
    } else {
        (
            u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize,
            10,
        )
    };
    let header_end = header_start
        .checked_add(header_len)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| bad("header extends past entry"))?;
    let header = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|_| bad("header is not UTF-8"))?;

    if find_flag(header, "'fortran_order'")? {
        return Err(Error::Unsupported(
            "npz: fortran_order arrays are refused (reversing the shape would silently \
             transpose the data)"
                .into(),
        ));
    }
    let descr = find_str(header, "'descr'").ok_or_else(|| bad("header missing 'descr'"))?;
    let leaf = map_descr(&descr)?;
    let shape = find_shape(header)?;

    Ok(NpyHeader {
        leaf,
        shape,
        data_offset: header_end,
    })
}

fn find_str(header: &str, key: &str) -> Option<String> {
    let after = header.split(key).nth(1)?.trim_start().strip_prefix(':')?;
    let t = after.trim_start();
    let quote = t.chars().next().filter(|&q| q == '\'' || q == '"')?;
    let inner = &t[1..];
    Some(inner[..inner.find(quote)?].to_string())
}

fn find_flag(header: &str, key: &str) -> Result<bool> {
    let Some(after) = header.split(key).nth(1) else {
        return Ok(false);
    };
    let after = after.trim_start();
    let after = after.strip_prefix(':').unwrap_or(after).trim_start();
    Ok(after.starts_with("True"))
}

fn find_shape(header: &str) -> Result<Vec<u64>> {
    let after = header
        .split("'shape'")
        .nth(1)
        .and_then(|s| s.trim_start().strip_prefix(':'))
        .ok_or_else(|| bad("header missing 'shape'"))?;
    let open = after.find('(').ok_or_else(|| bad("shape missing '('"))?;
    let close = after.find(')').ok_or_else(|| bad("shape missing ')'"))?;
    if close < open {
        return Err(bad("malformed shape tuple"));
    }
    let mut dims = Vec::new();
    for tok in after[open + 1..close].split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue; // "(3,)" or "()"
        }
        dims.push(
            tok.parse::<u64>()
                .map_err(|_| bad(format!("bad shape dim {tok:?}")))?,
        );
    }
    Ok(dims)
}

/// Where one entry's bytes are, before they become a payload.
enum Where {
    /// Stored entry: an absolute range of the file.
    Stored { offset: u64, length: u64 },
    /// Deflated entry: the zip entry to inflate, and how much of the result to
    /// drop (the `.npy` header) before the tensor starts.
    Deflated {
        zip_index: usize,
        data_offset: usize,
    },
}

/// Inflates deflated entries on demand. Keeps the archive open because the
/// bytes cannot be addressed, because there is nowhere to point at.
struct Deflated {
    archive: std::sync::Mutex<zip::ZipArchive<File>>,
    /// Keyed by the `key` in [`Payload::Opaque`]: (zip index, header length).
    entries: Vec<(usize, usize)>,
}

impl Decode for Deflated {
    fn decode(&self, key: u64, decoded_len: u64) -> Result<Vec<u8>> {
        let (zip_index, data_offset) = *self
            .entries
            .get(key as usize)
            .ok_or_else(|| bad(format!("no deflated entry {key}")))?;
        // The expected size is the *validated* one (shape x width plus the
        // header), never the ZIP's declared size: reading with a hard limit
        // keeps a lying entry from driving the allocation.
        let expected = crate::safe::add("npz entry", data_offset as u64, decoded_len)?;
        let cap = crate::safe::alloc_size("npz entry", expected)?;
        let mut archive = self.archive.lock().expect("npz archive lock");
        let mut entry = archive
            .by_index(zip_index)
            .map_err(|e| bad(format!("ZIP entry: {e}")))?;
        let mut bytes = Vec::with_capacity(cap);
        std::io::Read::take(&mut entry, expected + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| bad(format!("decompress: {e}")))?;
        if bytes.len() as u64 != expected {
            return Err(bad("decompressed size mismatch"));
        }
        Ok(bytes.split_off(data_offset))
    }
}

pub(crate) fn project(store: &Store) -> Result<Projection> {
    let mut archive = zip::ZipArchive::new(File::open(store.path())?)
        .map_err(|e| bad(format!("not a ZIP archive: {e}")))?;

    let mut catalog = Catalog::new();
    let mut deflated: Vec<(usize, usize)> = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| bad(format!("ZIP entry {i}: {e}")))?;
        let Some(name) = entry.name().strip_suffix(".npy").map(str::to_string) else {
            continue;
        };

        let stored = entry.compression() == zip::CompressionMethod::Stored;
        let (header, placed) = if stored {
            let start = entry.data_start();
            let size = entry.size();
            if crate::safe::add("npz entry", start, size)? > store.len() {
                return Err(bad(format!("entry {name:?} extends past file")));
            }
            // The header is at most a few hundred bytes; read what a v2 header
            // can reach rather than the whole entry.
            let probe = store.read(start, size.min(4096))?;
            let header = parse_npy_header(&probe)?;
            let data_off = crate::safe::add("npz entry", start, header.data_offset as u64)?;
            let length = size - header.data_offset as u64;
            (
                header,
                Where::Stored {
                    offset: data_off,
                    length,
                },
            )
        } else {
            // Parse only the header now; the payload inflates on demand.
            let mut head = vec![0u8; 4096.min(entry.size() as usize)];
            entry
                .read_exact(&mut head)
                .map_err(|e| bad(format!("entry {name:?}: {e}")))?;
            let header = parse_npy_header(&head)?;
            let data_offset = header.data_offset;
            (
                header,
                Where::Deflated {
                    zip_index: i,
                    data_offset,
                },
            )
        };
        let entry_size = entry.size();
        drop(entry);

        // Size equation: never trust the header blindly.
        let elems = crate::safe::product("npz shape", &header.shape)?;
        let expected = header
            .leaf
            .size(elems)
            .ok_or_else(|| bad("size not computable"))?;
        let actual = match &placed {
            Where::Stored { length, .. } => *length,
            Where::Deflated { data_offset, .. } => entry_size - *data_offset as u64,
        };
        if actual != expected {
            return Err(bad(format!(
                "entry {name:?} holds {actual} bytes but shape implies {expected}"
            )));
        }

        let payload = match placed {
            Where::Stored { offset, length } => Payload::At(Location {
                store: StoreId(0),
                offset,
                len: length,
            }),
            Where::Deflated {
                zip_index,
                data_offset,
            } => {
                deflated.push((zip_index, data_offset));
                Payload::Opaque {
                    store: StoreId(0),
                    key: deflated.len() as u64 - 1,
                    decoded_len: expected,
                }
            }
        };

        if catalog
            .insert(
                name.clone(),
                Entry {
                    shape: header.shape,
                    term: Some(header.leaf.into()),
                    layout: None,
                    attributes: None,
                    payload,
                    digest: None,
                    blocks: None,
                },
            )
            .is_some()
        {
            return Err(bad(format!("duplicate entry name {name:?}")));
        }
    }

    // A ZIP packs entries back to back behind local headers, so this file
    // cannot say which bytes are free, so occupancy stays unknown and page
    // exclusivity is never claimed.
    let projection = Projection::new(catalog);
    Ok(if deflated.is_empty() {
        projection
    } else {
        projection.with_decoder(Box::new(Deflated {
            archive: std::sync::Mutex::new(archive),
            entries: deflated,
        }))
    })
}
