//! A checkpoint's own files. [`read`] turns a snapshot directory into a
//! [`Metadata`], [`zt`] a single container, [`write`] puts one back on disk,
//! [`meta`] owns the reserved `__meta__/` names, and [`emit`]/[`serve`] are
//! the `pie.serving/1` writer/reader pair. The only place in the crate where
//! a path becomes a [`Metadata`].

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

/// What a checkpoint says about itself: a GGUF's key-value block. Read with
/// [`parse_attributes`](crate::file::read::parse_attributes) rather than
/// carried on [`Metadata`]. Flat, because GGUF's keys already are:
/// `general.architecture`, then a block namespaced under whatever that says.
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
    /// An array or a nested map, recorded as present and not as contents
    /// (e.g. GGUF's `tokenizer.ggml.tokens`, which no reader of this type
    /// wants in full). The key is kept so absence still means absent.
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
    /// `gemma3`. The one key that is not namespaced, because it is the key
    /// that says what the namespace is.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.text("general.architecture")
    }

    /// This key-value block as a flat JSON object, with dotted keys left
    /// dotted (`qwen2.block_count` stays one key, since the dots are part
    /// of GGUF's naming convention rather than real nesting).
    /// [`Attribute::Aggregate`] and a non-finite float both render as
    /// `null`.
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

/// GGUF's tokenizer tables, read whole. The one thing [`Attributes`]
/// deliberately does not carry (a vocabulary is 150,000 strings, kept as
/// [`Attribute::Aggregate`] there). Owned rather than borrowed since the
/// source is a CBOR tree inside a memory map the caller has no reason to
/// keep open.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenizerTables {
    /// `tokenizer.ggml.model` — `gpt2`, `llama`, `bert`, `rwkv`. Which
    /// FAMILY of tokenizer, not which model.
    pub model: String,
    /// `tokenizer.ggml.pre` — llama.cpp's name for a pre-tokenizer it has
    /// hard-coded, `qwen2` or `llama-bpe`. Absent in older files. A name,
    /// not a pattern: GGUF stores the identity where `tokenizer.json`
    /// stores regexes.
    pub pre: Option<String>,
    /// `tokenizer.ggml.tokens` — every token's text, in id order.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.token_type` — one per token, parallel to `tokens`.
    /// ggml's `llama_token_type`: 1 normal, 2 unknown, 3 control, 4 user
    /// defined, 5 unused, 6 byte.
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
    /// This is the enumeration a weight consumer wants, not `tensors`: pie
    /// stores its compiled tokenizer and model descriptor as `u8`
    /// objects indistinguishable from raw weights except by name.
    pub fn weights(&self) -> impl Iterator<Item = &RawTensor> {
        self.tensors
            .iter()
            .filter(|tensor| !meta::is_meta(&tensor.name))
    }

    /// The artifact's metadata objects, in manifest order. Empty for every
    /// checkpoint pie did not write.
    pub fn meta_objects(&self) -> impl Iterator<Item = &RawTensor> {
        self.tensors
            .iter()
            .filter(|tensor| meta::is_meta(&tensor.name))
    }

    /// The metadata object named `path` (without the [`meta::META_PREFIX`]).
    /// Its bytes come from [`read::read_meta`](crate::file::read::read_meta):
    /// this layer addresses, the reader opens.
    pub fn meta_object(&self, path: &str) -> Option<&RawTensor> {
        let name = meta::meta_name(path);
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// A checkpoint's tensors, indexed by name for one compile; the linear
/// `Metadata::tensor_by_name` makes a 32k-tensor compile quadratic. Indexes
/// weights only: a contract naming a metadata object ([`meta`]) fails to
/// resolve.
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

