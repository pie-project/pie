pub mod read;
pub mod write;
pub mod zt;

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

/// A checkpoint's tensors, indexed by name for the duration of one compile.
///
/// `tensor_by_name` is a linear scan, and both the resolver and the builder
/// call it once per contract tensor — which made compiling a 32k-tensor
/// checkpoint quadratic, and measurably so: 2.1 s, of which this was most.
///
/// The index lives here rather than on [`CheckpointMetadata`] because it is a
/// fact about a *compilation*, not about a checkpoint. Metadata is built by
/// readers, by tests and across the FFI boundary, and none of them should have
/// to carry a cache they never read.
pub struct Sources<'a> {
    metadata: &'a CheckpointMetadata,
    by_name: std::collections::HashMap<&'a str, u32>,
}

impl<'a> Sources<'a> {
    pub fn new(metadata: &'a CheckpointMetadata) -> Self {
        let by_name = metadata
            .tensors
            .iter()
            .enumerate()
            .filter_map(|(at, tensor)| u32::try_from(at).ok().map(|at| (tensor.name.as_str(), at)))
            .collect();
        Self { metadata, by_name }
    }

    pub fn metadata(&self) -> &'a CheckpointMetadata {
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

impl crate::contract::infer::CheckpointTypes for CheckpointMetadata {
    fn tensor_type(&self, name: &str) -> Option<crate::contract::TensorType> {
        self.tensor_by_name(name)
            .map(|raw| crate::contract::TensorType {
                shape: raw.shape.clone(),
                encoding: crate::types::normalize_encoding(&raw.encoding),
            })
    }
}
