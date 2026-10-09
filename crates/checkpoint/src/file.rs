pub mod diffusers;
pub mod emit;
pub mod meta;
pub mod read;
pub mod serve;
pub mod snapshot;
pub mod write;
pub mod zt;

pub use zt::encoding_of;

use crate::types::{CheckpointFormat, Encoding, FileId, TensorId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub files: Vec<File>,
    pub tensors: Vec<RawTensor>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attributes {
    by_key: std::collections::BTreeMap<String, Attribute>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
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

    #[must_use]
    pub fn text(&self, key: &str) -> Option<&str> {
        match self.by_key.get(key)? {
            Attribute::Text(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.text("general.architecture")
    }

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenizerTables {
    pub model: String,
    pub pre: Option<String>,
    pub tokens: Vec<String>,
    pub token_types: Vec<i64>,
    pub merges: Vec<String>,
}

impl TokenizerTables {
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

    pub fn weights(&self) -> impl Iterator<Item = &RawTensor> {
        self.tensors
            .iter()
            .filter(|tensor| !meta::is_meta(&tensor.name))
    }

    pub fn meta_objects(&self) -> impl Iterator<Item = &RawTensor> {
        self.tensors
            .iter()
            .filter(|tensor| meta::is_meta(&tensor.name))
    }

    pub fn meta_object(&self, path: &str) -> Option<&RawTensor> {
        let name = meta::meta_name(path);
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

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
        self.weights()
            .find(|tensor| tensor.name == name)
            .map(|raw| crate::contract::TensorType {
                shape: raw.shape.clone(),
                encoding: crate::types::normalize_encoding(&raw.encoding),
            })
    }
}
