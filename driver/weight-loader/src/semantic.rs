use serde::{Deserialize, Serialize};

use crate::types::{SemanticId, TensorId};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticGraph {
    pub tensors: Vec<SemanticTensor>,
    pub groups: Vec<SemanticGroup>,
}

impl SemanticGraph {
    pub fn empty() -> Self {
        Self {
            tensors: Vec::new(),
            groups: Vec::new(),
        }
    }

    pub fn tensor(&self, id: SemanticId) -> Option<&SemanticTensor> {
        self.tensors.iter().find(|tensor| tensor.id == id)
    }

    pub fn tensors_by_role(
        &self,
        role: &SemanticRole,
        layer: Option<u32>,
    ) -> impl Iterator<Item = &SemanticTensor> {
        self.tensors.iter().filter(move |tensor| {
            tensor.role == *role && layer.is_none_or(|layer| tensor.layer == Some(layer))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticTensor {
    pub id: SemanticId,
    pub role: SemanticRole,
    pub raw: TensorId,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticRole {
    TokenEmbedding,
    OutputEmbedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    MlpGate,
    MlpUp,
    MlpDown,
    ExpertRouter,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    ExpertBias,
    Norm,
    QuantData,
    QuantScale,
    QuantZeroPoint,
    QuantGroupIndex,
    Extension(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticGroupKind {
    AttentionQkv,
    MlpGateUp,
    ExpertBank,
    ExpertGateUpInterleaved,
    QuantizedTensor,
    GptOssMxfp4,
    Extension(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupMetadata {
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticGroup {
    pub kind: SemanticGroupKind,
    pub members: Vec<SemanticId>,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
    pub metadata: GroupMetadata,
}
