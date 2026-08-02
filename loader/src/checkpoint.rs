pub mod meta;
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
    /// Its *bytes* come from [`read::read_meta`](crate::checkpoint::read::read_meta):
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
/// The index lives here rather than on [`CheckpointMetadata`] because it is a
/// fact about a *compilation*, not about a checkpoint. Metadata is built by
/// readers, by tests and across the FFI boundary, and none of them should have
/// to carry a cache they never read.
///
/// It indexes weights only. A contract names the tensors a model family binds,
/// and pie's own metadata objects ([`meta`]) are not among them — so a contract
/// that names one fails to resolve, which is the reserved namespace being
/// reserved rather than a name that happens to be unused.
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
            .filter(|(_, tensor)| !meta::is_meta(&tensor.name))
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
        // Weights only, for the same reason `Sources` indexes weights only.
        self.weights()
            .find(|tensor| tensor.name == name)
            .map(|raw| crate::contract::TensorType {
                shape: raw.shape.clone(),
                encoding: crate::types::normalize_encoding(&raw.encoding),
            })
    }
}
