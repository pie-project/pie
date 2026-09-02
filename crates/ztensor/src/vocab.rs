//! L2 vocabulary: named layouts and encodings, held as a value.
//!
//! A profile is the code form of a registry mini-spec: implementable from its
//! text alone, and a reader that does not know one refuses to interpret rather
//! than guessing. The spec calls this layer registry-managed, so the registry
//! is a [`Vocabulary`] you can extend and hand to a reader or a writer:
//!
//! ```no_run
//! # use ztensor::{Source, Vocabulary};
//! # fn f(my_layout: impl ztensor::vocab::Layout + 'static) -> ztensor::Result<()> {
//! let vocab = Vocabulary::standard().with_layout(my_layout);
//! let src = Source::options().vocabulary(&vocab).open("model.zt")?;
//! # Ok(()) }
//! ```
//!
//! The canonical layout is not here: it is derived from the type (§5.1) and
//! every reader knows it. A layout in this registry is a departure from it.

use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result, Rule};
use crate::format::cbor::Value;
use crate::format::{align_up, Leaf, Object, Plane, Term, PLANE_ALIGN};

/// A named layout: how an object's bytes lie when they do not follow the
/// canonical rule (spec §5.2).
///
/// `validate` runs at open time (and at write time) on metadata only: the
/// type it admits, the attributes it needs, its size equation. Data-level
/// rules run when the object is actually assembled.
pub trait Layout: Send + Sync {
    fn id(&self) -> &str;
    fn validate(&self, name: &str, obj: &Object) -> Result<()>;
}

/// An encoding profile: a byte-stream transform for one blob.
pub trait Encoding: Send + Sync {
    fn id(&self) -> &str;
    fn encode(&self, decoded: &[u8]) -> Result<Vec<u8>>;
    /// Must produce exactly `decoded_length` bytes or reject.
    fn decode(&self, stored: &[u8], decoded_length: u64) -> Result<Vec<u8>>;
}

/// The set of profiles a reader or writer knows.
///
/// Later registrations shadow earlier ones, so a caller can replace a standard
/// profile as well as add to it.
#[derive(Clone, Default)]
pub struct Vocabulary {
    layouts: Vec<Arc<dyn Layout>>,
    encodings: Vec<Arc<dyn Encoding>>,
}

impl std::fmt::Debug for Vocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocabulary")
            .field(
                "layouts",
                &self.layouts.iter().map(|p| p.id()).collect::<Vec<_>>(),
            )
            .field(
                "encodings",
                &self.encodings.iter().map(|p| p.id()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Vocabulary {
    /// The profiles this implementation ships: `zt.sparse_csr/2`, the
    /// `gguf.<type>/2` family, and `zt.zstd-seekable/1` (with the `zstd`
    /// feature). `Default` knows nothing: every named layout is structural,
    /// every encoding undecodable.
    pub fn standard() -> Self {
        let mut v = Self::default().with_layout(SparseCsr);
        for row in gguf::TABLE {
            v = v.with_layout(gguf::Gguf::new(row));
        }
        #[cfg(feature = "zstd")]
        {
            v = v.with_encoding(zstd_seekable::ZstdSeekable);
        }
        v
    }

    pub(crate) fn shared() -> Arc<Vocabulary> {
        static STANDARD: OnceLock<Arc<Vocabulary>> = OnceLock::new();
        STANDARD
            .get_or_init(|| Arc::new(Vocabulary::standard()))
            .clone()
    }

    pub fn with_layout(mut self, profile: impl Layout + 'static) -> Self {
        self.layouts.push(Arc::new(profile));
        self
    }

    pub fn with_encoding(mut self, profile: impl Encoding + 'static) -> Self {
        self.encodings.push(Arc::new(profile));
        self
    }

    /// `None` means structural-only access: the object is readable as bytes,
    /// its layout rules unchecked.
    pub fn layout(&self, id: &str) -> Option<&dyn Layout> {
        self.layouts
            .iter()
            .rev()
            .find(|p| p.id() == id)
            .map(Arc::as_ref)
    }

    /// `None` means the stored bytes can be addressed but not decoded.
    pub fn encoding(&self, id: &str) -> Option<&dyn Encoding> {
        self.encodings
            .iter()
            .rev()
            .find(|p| p.id() == id)
            .map(Arc::as_ref)
    }
}

fn attr_u64(attributes: Option<&Value>, key: &str) -> Option<u64> {
    attributes?.get(key)?.as_u64()
}

fn attr_text<'a>(attributes: Option<&'a Value>, key: &str) -> Option<&'a str> {
    attributes?.get(key)?.as_text()
}

// =======================================================================
// zt.sparse_csr/2 (spec/profiles/zt.sparse_csr-2.md)
// =======================================================================

/// The byte plan of a `zt.sparse_csr/2` blob: `indptr`, then `indices`, then
/// the value planes, each at the next plane boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrPlan {
    pub rows: u64,
    pub cols: u64,
    pub nnz: u64,
    pub index: Leaf,
    pub indptr: Plane,
    pub indices: Plane,
    /// The value planes, offsets shifted past the index planes.
    pub values: Vec<Plane>,
    pub size: u64,
}

impl CsrPlan {
    /// Reads the plan off an object's metadata, checking the profile's
    /// metadata rules.
    pub fn of(
        name: &str,
        shape: &[u64],
        term: Option<&Term>,
        attributes: Option<&Value>,
    ) -> Result<CsrPlan> {
        let fail = |detail: String| Err(Error::reject(Rule::LayoutRule, format!("{name:?}: {detail}")));
        let [rows, cols] = shape[..] else {
            return fail("sparse_csr requires rank-2 shape".into());
        };
        let Some(term) = term else {
            return fail("sparse_csr requires a type for its values".into());
        };
        let Some(nnz) = attr_u64(attributes, "nnz") else {
            return fail("sparse_csr requires attribute 'nnz'".into());
        };
        let index = match attr_text(attributes, "index") {
            Some("u32") => Leaf::U32,
            Some("u64") => Leaf::U64,
            _ => return fail("sparse_csr requires attribute 'index' of \"u32\" or \"u64\"".into()),
        };
        let overflow = || Error::reject(Rule::Shape, format!("{name:?}: size overflows u64"));
        let indptr_len = rows
            .checked_add(1)
            .and_then(|n| index.size(n))
            .ok_or_else(overflow)?;
        let indices_at = align_up(indptr_len, PLANE_ALIGN).ok_or_else(overflow)?;
        let indices_len = index.size(nnz).ok_or_else(overflow)?;
        let values_at = indices_at
            .checked_add(indices_len)
            .and_then(|end| align_up(end, PLANE_ALIGN))
            .ok_or_else(overflow)?;
        let mut values = term.planes(&[nnz]).map_err(|e| e.at(name))?;
        for p in &mut values {
            p.offset = p.offset.checked_add(values_at).ok_or_else(overflow)?;
        }
        let size = match values.last() {
            Some(p) => p.offset.checked_add(p.len).ok_or_else(overflow)?,
            None => values_at,
        };
        Ok(CsrPlan {
            rows,
            cols,
            nnz,
            index,
            indptr: Plane {
                path: "indptr".into(),
                leaf: index,
                shape: vec![rows + 1],
                offset: 0,
                len: indptr_len,
            },
            indices: Plane {
                path: "indices".into(),
                leaf: index,
                shape: vec![nnz],
                offset: indices_at,
                len: indices_len,
            },
            values,
            size,
        })
    }
}

struct SparseCsr;

impl Layout for SparseCsr {
    fn id(&self) -> &str {
        "zt.sparse_csr/2"
    }

    fn validate(&self, name: &str, obj: &Object) -> Result<()> {
        let plan = CsrPlan::of(name, &obj.shape, obj.term.as_ref(), obj.attributes.as_ref())?;
        if obj.blob.decoded_size() != plan.size {
            return Err(Error::reject(
                Rule::Size,
                format!(
                    "{name:?}: decoded size {} != the {} sparse_csr requires",
                    obj.blob.decoded_size(),
                    plan.size
                ),
            ));
        }
        Ok(())
    }
}

// =======================================================================
// gguf.<type>/2 (spec/profiles/gguf.md)
// =======================================================================

pub mod gguf {
    //! ggml block formats, kept byte for byte. Each row of [`TABLE`] is one
    //! `gguf.<type>/2` layout: its block geometry and the term its values
    //! have, or `None` for the codebook types no term expresses.

    use super::*;

    pub struct Row {
        pub name: &'static str,
        pub elems_per_block: u64,
        pub block_bytes: u64,
        pub term: Option<&'static str>,
    }

    const fn row(name: &'static str, epb: u64, bytes: u64, term: Option<&'static str>) -> Row {
        Row {
            name,
            elems_per_block: epb,
            block_bytes: bytes,
            term,
        }
    }

    pub const TABLE: &[Row] = &[
        row("q4_0", 32, 18, Some("g32_i4_f16_n")),
        row("q4_1", 32, 20, Some("g32_u4_f16_b_f16")),
        row("q5_0", 32, 22, Some("g32_i5_f16_n")),
        row("q5_1", 32, 24, Some("g32_u5_f16_b_f16")),
        row("q8_0", 32, 34, Some("g32_i8_f16_n")),
        row("q8_1", 32, 36, Some("g32_i8_f16_n")),
        row("q2_k", 256, 84, Some("g16_u2_g16_u4_f16_n_b_g16_u4_f16_n")),
        row("q3_k", 256, 110, Some("g16_i3_g16_i6_f16_n_n")),
        row("q4_k", 256, 144, Some("g32_u4_g8_u6_f16_n_b_g8_u6_f16_n")),
        row("q5_k", 256, 176, Some("g32_u5_g8_u6_f16_n_b_g8_u6_f16_n")),
        row("q6_k", 256, 210, Some("g16_i6_g16_i8_f16_n_n")),
        row("q8_k", 256, 292, Some("g256_i8_f32_n")),
        row("iq1_s", 256, 50, None),
        row("iq1_m", 256, 56, None),
        row("iq2_xxs", 256, 66, None),
        row("iq2_xs", 256, 74, None),
        row("iq2_s", 256, 82, None),
        row("iq3_xxs", 256, 98, None),
        row("iq3_s", 256, 110, None),
        row("iq4_nl", 32, 18, None),
        row("iq4_xs", 256, 136, None),
        row("mxfp4", 32, 17, Some("g32_e2m1_e8m0_n")),
    ];

    /// The row for a ggml type name.
    pub fn row_of(name: &str) -> Option<&'static Row> {
        TABLE.iter().find(|r| r.name == name)
    }

    impl Row {
        pub fn layout_id(&self) -> String {
            format!("gguf.{}/2", self.name)
        }

        pub fn term(&self) -> Option<Term> {
            self.term.map(|t| Term::parse(t).expect("the table spells terms correctly"))
        }
    }

    pub(super) struct Gguf {
        row: &'static Row,
        id: String,
        term: Option<Term>,
    }

    impl Gguf {
        pub(super) fn new(row: &'static Row) -> Self {
            Gguf {
                row,
                id: row.layout_id(),
                term: row.term(),
            }
        }
    }

    impl Layout for Gguf {
        fn id(&self) -> &str {
            &self.id
        }

        fn validate(&self, name: &str, obj: &Object) -> Result<()> {
            let row = self.row;
            let fail = |detail: String| {
                Err(Error::reject(Rule::LayoutRule, format!("{name:?}: {detail}")))
            };
            match (&obj.term, &self.term) {
                (None, None) => {}
                (Some(got), Some(want)) if got == want => {}
                (Some(got), Some(want)) => {
                    return fail(format!("{} holds {want}, not {got}", row.layout_id()))
                }
                (Some(_), None) => {
                    return fail(format!("{} is a codebook format and takes no type", row.layout_id()))
                }
                (None, Some(want)) => return fail(format!("{} requires type {want}", row.layout_id())),
            }
            let attributes = obj.attributes.as_ref();
            if attr_u64(attributes, "elems_per_block") != Some(row.elems_per_block)
                || attr_u64(attributes, "block_bytes") != Some(row.block_bytes)
            {
                return fail(format!(
                    "{} requires elems_per_block {} and block_bytes {}",
                    row.layout_id(),
                    row.elems_per_block,
                    row.block_bytes
                ));
            }
            let fastest = obj.shape.last().copied().unwrap_or(1);
            if fastest % row.elems_per_block != 0 {
                return fail(format!(
                    "fastest axis {fastest} is not a multiple of {} elements per block",
                    row.elems_per_block
                ));
            }
            let blocks = obj.num_elements()? / row.elems_per_block;
            let expected = blocks
                .checked_mul(row.block_bytes)
                .ok_or_else(|| Error::reject(Rule::Shape, format!("{name:?}: size overflows u64")))?;
            if obj.blob.decoded_size() != expected {
                return Err(Error::reject(
                    Rule::Size,
                    format!(
                        "{name:?}: decoded size {} != {expected} ({blocks} blocks of {})",
                        obj.blob.decoded_size(),
                        row.block_bytes
                    ),
                ));
            }
            Ok(())
        }
    }
}

// =======================================================================
// zt.zstd-seekable/1 (spec/profiles/zt.zstd-seekable-1.md)
// =======================================================================

#[cfg(feature = "zstd")]
mod zstd_seekable {
    use std::io::Write;

    use super::Encoding;
    use crate::error::{Error, Result, Rule};

    /// Decoded bytes per frame. Spec: ≤ 16 MiB, all frames equal-sized
    /// except the last.
    const CHUNK: usize = 1 << 20;
    const MAX_FRAME: u64 = 16 << 20;
    const LEVEL: i32 = 3;
    const SKIPPABLE_MAGIC: u32 = 0x184D2A5E;
    const SEEKABLE_MAGIC: u32 = 0x8F92EAB1;

    pub struct ZstdSeekable;

    fn bad(detail: impl Into<String>) -> Error {
        Error::reject(Rule::Encoding, detail.into())
    }

    impl Encoding for ZstdSeekable {
        fn id(&self) -> &str {
            "zt.zstd-seekable/1"
        }

        fn encode(&self, decoded: &[u8]) -> Result<Vec<u8>> {
            let mut out = Vec::new();
            let mut entries: Vec<(u32, u32)> = Vec::new();
            for chunk in decoded.chunks(CHUNK) {
                let mut enc = zstd::stream::write::Encoder::new(Vec::new(), LEVEL)?;
                enc.include_checksum(true)?;
                enc.write_all(chunk)?;
                let frame = enc.finish()?;
                entries.push((frame.len() as u32, chunk.len() as u32));
                out.extend_from_slice(&frame);
            }
            let content_len = entries.len() * 8 + 9;
            out.extend(SKIPPABLE_MAGIC.to_le_bytes());
            out.extend((content_len as u32).to_le_bytes());
            for (c, d) in &entries {
                out.extend(c.to_le_bytes());
                out.extend(d.to_le_bytes());
            }
            out.extend((entries.len() as u32).to_le_bytes());
            out.push(0u8);
            out.extend(SEEKABLE_MAGIC.to_le_bytes());
            Ok(out)
        }

        fn decode(&self, stored: &[u8], decoded_length: u64) -> Result<Vec<u8>> {
            let n = stored.len();
            if n < 17 {
                return Err(bad("stream too short for a seek table"));
            }
            if stored[n - 4..] != SEEKABLE_MAGIC.to_le_bytes() {
                return Err(bad("missing seekable footer magic"));
            }
            let descriptor = stored[n - 5];
            if descriptor & 0x7f != 0 {
                return Err(bad("reserved descriptor bits set"));
            }
            let entry_size = if descriptor & 0x80 != 0 { 12 } else { 8 };
            let frames_n = u32::from_le_bytes(stored[n - 9..n - 5].try_into().unwrap()) as usize;
            let content_len = frames_n
                .checked_mul(entry_size)
                .and_then(|v| v.checked_add(9))
                .ok_or_else(|| bad("seek table size overflow"))?;
            let table_start = n
                .checked_sub(8 + content_len)
                .ok_or_else(|| bad("seek table larger than stream"))?;
            if stored[table_start..table_start + 4] != SKIPPABLE_MAGIC.to_le_bytes()
                || stored[table_start + 4..table_start + 8] != (content_len as u32).to_le_bytes()
            {
                return Err(bad("malformed seek table skippable frame"));
            }

            let mut expected: Vec<(u64, u64)> = Vec::with_capacity(frames_n);
            let mut pos = table_start + 8;
            let mut total_c = 0u64;
            let mut total_d = 0u64;
            for _ in 0..frames_n {
                let c = u32::from_le_bytes(stored[pos..pos + 4].try_into().unwrap()) as u64;
                let d = u32::from_le_bytes(stored[pos + 4..pos + 8].try_into().unwrap()) as u64;
                pos += entry_size;
                if d > MAX_FRAME {
                    return Err(bad("frame content exceeds 16 MiB"));
                }
                total_c += c;
                total_d += d;
                expected.push((c, d));
            }
            if total_d != decoded_length {
                return Err(bad(format!(
                    "seek table totals {total_d} bytes, declared decoded_length {decoded_length}"
                )));
            }
            if total_c != table_start as u64 {
                return Err(bad("frame sizes do not cover the stream"));
            }
            if let Some((_, first_d)) = expected.first() {
                for (_, d) in &expected[..expected.len() - 1] {
                    if d != first_d {
                        return Err(bad("non-final frames must be equal-sized"));
                    }
                }
            }

            let mut out = Vec::with_capacity(decoded_length as usize);
            let mut fpos = 0usize;
            for (c, d) in expected {
                let frame = &stored[fpos..fpos + c as usize];
                let decoded = zstd::bulk::decompress(frame, d as usize)
                    .map_err(|e| bad(format!("frame decompression failed: {e}")))?;
                if decoded.len() as u64 != d {
                    return Err(bad("frame decoded to a different size than declared"));
                }
                out.extend_from_slice(&decoded);
                fpos += c as usize;
            }
            Ok(out)
        }
    }
}
