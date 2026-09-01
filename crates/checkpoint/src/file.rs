//! A checkpoint's own files: reading them, writing them, and the namespace
//! pie reserves inside one.
//!
//! Six modules over one subject. [`read`] turns a snapshot *directory* into a
//! [`Metadata`], [`zt`] turns a single *container* into one, [`write`] puts
//! one back on disk, and [`meta`] owns the `__meta__/` names that tell a pie
//! artifact's own payloads apart from its weights. [`emit`] and [`serve`] are
//! the `pie.serving/1` pair — the import-only writer and the lean serving
//! reader — and both spell their agreement in [`serving`](crate::serving),
//! which sits outside this directory because it opens nothing.
//!
//! Everything above this module computes over the value it produces: this is
//! the only place in the crate where a *path* becomes one,
//! which is what `tests/standalone.rs` holds the compiler to. The two files it
//! exempts beside this directory — `executor/walk.rs` and `verify.rs` — copy
//! bytes and compare a plan against the world; neither decides anything.
//!
//! # It was called `checkpoint`, inside a crate called `checkpoint`
//!
//! And a consumer wrote `checkpoint::checkpoint::read::parse_checkpoint_metadata`,
//! which says one word three times and says what the thing IS once. The crate
//! already names the subject; the module names the surface — a checkpoint's
//! files — and the items in it drop the prefix for the same reason, the one
//! that makes `io::Error` right and `io::IoError` wrong. [`Metadata`] is the
//! checkpoint's, because there is nowhere else it could be from.

pub mod emit;
pub mod meta;
pub mod read;
pub mod serve;
pub mod write;
pub mod zt;

pub use zt::encoding_of;

use crate::types::{CheckpointFormat, Encoding, FileId, TensorId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub files: Vec<File>,
    pub tensors: Vec<RawTensor>,
}

/// What a checkpoint says about ITSELF: a GGUF's key-value block.
///
/// Read with [`parse_attributes`](crate::file::read::parse_attributes) rather
/// than carried on [`Metadata`] — see that function for why.
///
/// Flat, because GGUF's keys already are: `general.architecture`, and then a
/// block namespaced under whatever that says, `qwen2.attention.head_count`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attributes {
    by_key: std::collections::BTreeMap<String, Attribute>,
}

/// One key's value, as far as this type carries it.
#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    /// An array or a nested map, recorded as present and not as contents.
    ///
    /// GGUF puts its tokenizer here — `tokenizer.ggml.tokens` is every token
    /// in the vocabulary — and no reader of this type wants a vocabulary. The
    /// key is kept so that absence still means absent, which is the whole
    /// value of an honest answer to `get`.
    Aggregate,
}

impl Attributes {
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, Attribute)>) -> Self {
        Self {
            by_key: pairs.into_iter().collect(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Attribute> {
        self.by_key.get(key)
    }

    /// The value of `key`, when it is text.
    #[must_use]
    pub fn text(&self, key: &str) -> Option<&str> {
        match self.by_key.get(key)? {
            Attribute::Text(value) => Some(value),
            _ => None,
        }
    }

    /// Which architecture llama.cpp wrote this file for — `llama`, `qwen2`,
    /// `gemma3`.
    ///
    /// The one key that is not namespaced, because it is the key that says
    /// what the namespace IS. Everything the file states about its geometry
    /// hangs off this answer, and a converter that had to guess an
    /// architecture from tensor names would be guessing at exactly the point
    /// where being wrong is silent.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.text("general.architecture")
    }

    /// This key-value block as a flat JSON object.
    ///
    /// Flat, and with the dotted keys left dotted: `qwen2.block_count` stays
    /// one key rather than becoming `{"qwen2": {"block_count": ...}}`. The
    /// dots are part of the name in GGUF — `general.alignment` and
    /// `qwen2.attention.head_count` are namespaced by convention and not by
    /// structure — and inventing a nesting the format does not have would
    /// make the round trip lossy for any key whose prefix is also a key.
    ///
    /// [`Attribute::Aggregate`] renders as `null`, which is the same claim
    /// the variant makes: the key was there, the contents were not carried.
    /// A non-finite float renders as `null` for the same reason — JSON has
    /// no spelling for it, and a silent 0.0 would be a different number.
    #[must_use]
    pub fn to_json(&self) -> String {
        let map: serde_json::Map<String, serde_json::Value> = self
            .by_key
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    Attribute::Uint(n) => (*n).into(),
                    Attribute::Int(n) => (*n).into(),
                    Attribute::Float(n) => serde_json::Number::from_f64(*n)
                        .map_or(serde_json::Value::Null, serde_json::Value::Number),
                    Attribute::Bool(b) => (*b).into(),
                    Attribute::Text(s) => s.clone().into(),
                    Attribute::Aggregate => serde_json::Value::Null,
                };
                (key.clone(), value)
            })
            .collect();
        serde_json::Value::Object(map).to_string()
    }
}

/// GGUF's tokenizer tables, read whole.
///
/// The one thing [`Attributes`] deliberately does not carry. A vocabulary is
/// 150,000 strings and no reader of a model's DESCRIPTION wants it, so the
/// summary keeps `tokenizer.ggml.tokens` as an [`Attribute::Aggregate`] and
/// the caller that actually wants the contents asks for them here, by name
/// and on purpose.
///
/// Owned rather than borrowed because the source is a CBOR tree inside a
/// memory map that the caller has no reason to keep open, and owned rather
/// than compiled because compiling a tokenizer is the tokenizer crate's job
/// — this crate reads files.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenizerTables {
    /// `tokenizer.ggml.model` — `gpt2`, `llama`, `bert`, `rwkv`. Which
    /// FAMILY of tokenizer, not which model.
    pub model: String,
    /// `tokenizer.ggml.pre` — llama.cpp's name for a pre-tokenizer it has
    /// hard-coded, `qwen2` or `llama-bpe`. Absent in older files.
    ///
    /// A name and not a pattern, which is the whole difficulty: GGUF stores
    /// the IDENTITY of a pre-tokenizer where `tokenizer.json` stores its
    /// regexes, so a reader either knows the name or cannot proceed.
    pub pre: Option<String>,
    /// `tokenizer.ggml.tokens` — every token's text, in id order.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.token_type` — one per token, parallel to `tokens`.
    ///
    /// ggml's `llama_token_type`: 1 normal, 2 unknown, 3 control, 4 user
    /// defined, 5 unused, 6 byte. This is how a GGUF says which tokens are
    /// the added ones, a fact `tokenizer.json` keeps in a separate
    /// `added_tokens` list.
    pub token_types: Vec<i64>,
    /// `tokenizer.ggml.merges` — `"left right"`, in rank order.
    pub merges: Vec<String>,
}

impl TokenizerTables {
    /// Whether the file carried a tokenizer at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct File {
    pub id: FileId,
    pub path: String,
    pub size_bytes: u64,
    pub format: CheckpointFormat,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawTensor {
    pub id: TensorId,
    pub name: String,
    pub file_id: FileId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
}

impl Metadata {
    pub fn tensor(&self, id: TensorId) -> Option<&RawTensor> {
        self.tensors
            .get(id.0 as usize)
            .filter(|tensor| tensor.id == id)
            .or_else(|| self.tensors.iter().find(|tensor| tensor.id == id))
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// The checkpoint's weights — every object except pie's own metadata.
    ///
    /// **This is the enumeration a weight consumer wants**, not `tensors`. A
    /// pie artifact stores its compiled tokenizer and model descriptor as
    /// `dense` `u8` objects (zTensor has no non-tensor object, see
    /// [`meta`]), which are indistinguishable from raw `u8` weights except by
    /// name. Iterating `tensors` therefore plans, copies or uploads them; this
    /// does not. `tensors` remains public because a reader, a writer and the
    /// FFI marshaller each need the whole object list — but nothing that means
    /// "the model's weights" should reach for it.
    pub fn weights(&self) -> impl Iterator<Item = &RawTensor> {
        self.tensors
            .iter()
            .filter(|tensor| !meta::is_meta(&tensor.name))
    }

    /// The artifact's metadata objects, in manifest order.
    ///
    /// Empty for every checkpoint pie did not write.
    pub fn meta_objects(&self) -> impl Iterator<Item = &RawTensor> {
        self.tensors
            .iter()
            .filter(|tensor| meta::is_meta(&tensor.name))
    }

    /// The metadata object named `path` (without the [`meta::META_PREFIX`]).
    ///
    /// Its *bytes* come from [`read::read_meta`](crate::file::read::read_meta):
    /// this layer addresses, the reader opens.
    pub fn meta_object(&self, path: &str) -> Option<&RawTensor> {
        let name = meta::meta_name(path);
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// A checkpoint's tensors, indexed by name for the duration of one compile.
///
/// `tensor_by_name` is a linear scan, and both the resolver and the builder
/// call it once per contract tensor — which made compiling a 32k-tensor
/// checkpoint quadratic, and measurably so: 2.1 s, of which this was most.
///
/// The index lives here rather than on [`Metadata`] because it is a fact about
/// a *compilation*, not about a checkpoint. Metadata is built by
/// readers, by tests and across the FFI boundary, and none of them should have
/// to carry a cache they never read.
///
/// It indexes weights only. A contract names the tensors a model family binds,
/// and pie's own metadata objects ([`meta`]) are not among them — so a contract
/// that names one fails to resolve, which is the reserved namespace being
/// reserved rather than a name that happens to be unused.
pub struct Sources<'a> {
    metadata: &'a Metadata,
    by_name: std::collections::HashMap<&'a str, u32>,
}

impl<'a> Sources<'a> {
    pub fn new(metadata: &'a Metadata) -> Self {
        let by_name = metadata
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, tensor)| !meta::is_meta(&tensor.name))
            .filter_map(|(at, tensor)| u32::try_from(at).ok().map(|at| (tensor.name.as_str(), at)))
            .collect();
        Self { metadata, by_name }
    }

    pub fn metadata(&self) -> &'a Metadata {
        self.metadata
    }

    pub fn by_name(&self, name: &str) -> Option<&'a RawTensor> {
        self.metadata.tensors.get(*self.by_name.get(name)? as usize)
    }

    pub fn tensor(&self, id: TensorId) -> Option<&'a RawTensor> {
        self.metadata.tensor(id)
    }
}

impl crate::contract::infer::CheckpointTypes for Sources<'_> {
    fn tensor_type(&self, name: &str) -> Option<crate::contract::TensorType> {
        self.by_name(name).map(|raw| crate::contract::TensorType {
            shape: raw.shape.clone(),
            encoding: crate::types::normalize_encoding(&raw.encoding),
        })
    }
}

impl crate::contract::infer::CheckpointTypes for Metadata {
    fn tensor_type(&self, name: &str) -> Option<crate::contract::TensorType> {
        // Weights only, for the same reason `Sources` indexes weights only.
        self.weights()
            .find(|tensor| tensor.name == name)
            .map(|raw| crate::contract::TensorType {
                shape: raw.shape.clone(),
                encoding: crate::types::normalize_encoding(&raw.encoding),
            })
    }
}

#[cfg(test)]
mod attribute_tests {
    use super::{Attribute, Attributes};

    /// The rendering is what a GGUF import carries as `model/config`, so
    /// every variant has to survive it -- and the two that JSON cannot spell
    /// have to say so rather than pick a number.
    #[test]
    fn every_attribute_renders_and_the_unspellable_ones_say_null() {
        let attributes = Attributes::from_pairs([
            (
                "general.architecture".into(),
                Attribute::Text("qwen2".into()),
            ),
            ("qwen2.block_count".into(), Attribute::Uint(24)),
            ("a.negative".into(), Attribute::Int(-3)),
            ("a.float".into(), Attribute::Float(0.5)),
            ("a.bool".into(), Attribute::Bool(true)),
            ("a.nan".into(), Attribute::Float(f64::NAN)),
            ("tokenizer.ggml.tokens".into(), Attribute::Aggregate),
        ]);
        let json: serde_json::Value = serde_json::from_str(&attributes.to_json()).unwrap();

        assert_eq!(json["general.architecture"], "qwen2");
        assert_eq!(json["qwen2.block_count"], 24);
        assert_eq!(json["a.negative"], -3);
        assert_eq!(json["a.float"], 0.5);
        assert_eq!(json["a.bool"], true);
        // Present, and honest about carrying no value: a 0.0 here would be a
        // different number, and an omitted key would be a different fact.
        assert!(json["a.nan"].is_null());
        assert!(json["tokenizer.ggml.tokens"].is_null());
        assert!(
            json.as_object()
                .unwrap()
                .contains_key("tokenizer.ggml.tokens")
        );
    }

    /// The dots are part of the key, not a path. `general` is not an object
    /// here, and `qwen2.attention.head_count` is one key and not three.
    #[test]
    fn a_dotted_key_stays_one_key() {
        let attributes =
            Attributes::from_pairs([("qwen2.attention.head_count".into(), Attribute::Uint(14))]);
        let json: serde_json::Value = serde_json::from_str(&attributes.to_json()).unwrap();

        assert_eq!(json["qwen2.attention.head_count"], 14);
        assert!(json.get("qwen2").is_none());
    }

    /// The whole point of carrying it: `Encoding::from_config_json` has to
    /// read this document as an unquantized checkpoint. It IS one -- import
    /// decodes every GGUF block on the way in, because no device capability
    /// mask carries `DECODE` -- and a GGUF states its per-tensor scheme in
    /// the tensor record rather than in a `quantization_config` block, so
    /// there is nothing here for the reader to mistake for one.
    #[test]
    fn a_gguf_key_value_block_declares_no_quantization() {
        let attributes = Attributes::from_pairs([
            (
                "general.architecture".into(),
                Attribute::Text("qwen2".into()),
            ),
            ("general.file_type".into(), Attribute::Uint(2)),
            ("qwen2.block_count".into(), Attribute::Uint(24)),
        ]);
        let json: serde_json::Value = serde_json::from_str(&attributes.to_json()).unwrap();

        for key in ["quantization_config", "quantization", "text_config"] {
            assert!(json.get(key).is_none(), "{key} would change the reading");
        }
    }
}
