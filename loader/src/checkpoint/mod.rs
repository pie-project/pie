pub mod gguf;
pub mod header;

use crate::types::{CheckpointFormat, Encoding, FileId, TensorId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointMetadata {
    pub files: Vec<CheckpointFile>,
    pub tensors: Vec<RawTensor>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointFile {
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

impl CheckpointMetadata {
    pub fn tensor(&self, id: TensorId) -> Option<&RawTensor> {
        self.tensors
            .get(id.0 as usize)
            .filter(|tensor| tensor.id == id)
            .or_else(|| self.tensors.iter().find(|tensor| tensor.id == id))
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

impl crate::contract::CheckpointTypes for CheckpointMetadata {
    fn tensor_type(&self, name: &str) -> Option<crate::contract::TensorType> {
        self.tensor_by_name(name)
            .map(|raw| crate::contract::TensorType {
                shape: raw.shape.clone(),
                encoding: crate::types::normalize_encoding(&raw.encoding),
            })
    }
}
