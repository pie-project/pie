use std::collections::HashSet;

use crate::config::ModelConfig;
use crate::error::CompileError;
use crate::schema::ModelSchema;
use crate::semantic::{
    GroupMetadata, SemanticGraph, SemanticGroup, SemanticGroupKind, SemanticRole, SemanticTensor,
};
use crate::source::{CheckpointMetadata, RawTensor};
use crate::types::SemanticId;

pub struct GenericSchema {
    pub names: &'static [&'static str],
}

impl ModelSchema for GenericSchema {
    fn matches(&self, model_type: &str) -> bool {
        let lowered = model_type.to_ascii_lowercase();
        self.names
            .iter()
            .any(|name| lowered.contains(&name.to_ascii_lowercase()))
    }

    fn build(
        &self,
        metadata: &CheckpointMetadata,
        _cfg: &ModelConfig,
    ) -> Result<SemanticGraph, CompileError> {
        let mut graph = SemanticGraph::empty();
        for (index, raw) in metadata.tensors.iter().enumerate() {
            graph.tensors.push(SemanticTensor {
                id: SemanticId(index as u32),
                role: infer_role(raw),
                raw: raw.id,
                layer: infer_layer(&raw.name),
                expert: infer_expert(&raw.name),
            });
        }
        add_groups(&mut graph, metadata);
        Ok(graph)
    }
}

fn infer_role(raw: &RawTensor) -> SemanticRole {
    let name = raw.name.as_str();
    if name.ends_with("embed_tokens.weight") || name.ends_with("wte.weight") {
        SemanticRole::TokenEmbedding
    } else if name.ends_with("lm_head.weight") {
        SemanticRole::OutputEmbedding
    } else if contains_any(name, &["q_proj.weight", "wq.weight"]) {
        SemanticRole::AttentionQ
    } else if contains_any(name, &["k_proj.weight", "wk.weight"]) {
        SemanticRole::AttentionK
    } else if contains_any(name, &["v_proj.weight", "wv.weight"]) {
        SemanticRole::AttentionV
    } else if contains_any(name, &["o_proj.weight", "wo.weight"]) {
        SemanticRole::AttentionO
    } else if contains_any(name, &["gate_proj.weight", "w1.weight"]) {
        SemanticRole::MlpGate
    } else if contains_any(name, &["up_proj.weight", "w3.weight"]) {
        SemanticRole::MlpUp
    } else if contains_any(name, &["down_proj.weight", "w2.weight"]) {
        SemanticRole::MlpDown
    } else if contains_any(name, &["router.weight", "gate.weight"]) {
        SemanticRole::ExpertRouter
    } else if name.contains(".experts.") && contains_any(name, &["w1", "gate_proj"]) {
        SemanticRole::ExpertGate
    } else if name.contains(".experts.") && contains_any(name, &["w3", "up_proj"]) {
        SemanticRole::ExpertUp
    } else if name.contains(".experts.") && contains_any(name, &["w2", "down_proj"]) {
        SemanticRole::ExpertDown
    } else if name.ends_with(".bias") && name.contains(".experts.") {
        SemanticRole::ExpertBias
    } else if contains_any(name, &["norm.weight", "ln_f.weight", "layernorm.weight"]) {
        SemanticRole::Norm
    } else if contains_any(name, &["scale", "scales", "absmax", "d_scale"]) {
        SemanticRole::QuantScale
    } else if contains_any(name, &["zero", "zeros", "qzeros"]) {
        SemanticRole::QuantZeroPoint
    } else if name.ends_with("g_idx") || name.contains(".g_idx") {
        SemanticRole::QuantGroupIndex
    } else {
        SemanticRole::Extension(name.to_owned())
    }
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn infer_layer(name: &str) -> Option<u32> {
    parse_index_after(name, ".layers.")
        .or_else(|| parse_index_after(name, ".h."))
        .or_else(|| parse_index_after(name, "blk."))
}

fn infer_expert(name: &str) -> Option<u32> {
    parse_index_after(name, ".experts.").or_else(|| parse_index_after(name, ".block_sparse_moe."))
}

fn parse_index_after(name: &str, marker: &str) -> Option<u32> {
    let (_, suffix) = name.split_once(marker)?;
    let digits: String = suffix
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn add_groups(graph: &mut SemanticGraph, metadata: &CheckpointMetadata) {
    let layers: HashSet<u32> = graph
        .tensors
        .iter()
        .filter_map(|tensor| tensor.layer)
        .collect();
    for layer in layers {
        let qkv = members_for_roles(
            graph,
            layer,
            &[
                SemanticRole::AttentionQ,
                SemanticRole::AttentionK,
                SemanticRole::AttentionV,
            ],
        );
        if qkv.len() == 3 {
            graph.groups.push(SemanticGroup {
                kind: SemanticGroupKind::AttentionQkv,
                members: qkv,
                layer: Some(layer),
                expert: None,
                metadata: GroupMetadata {
                    name: format!("layer.{layer}.attention_qkv"),
                },
            });
        }
        let gate_up =
            members_for_roles(graph, layer, &[SemanticRole::MlpGate, SemanticRole::MlpUp]);
        if gate_up.len() == 2 {
            graph.groups.push(SemanticGroup {
                kind: SemanticGroupKind::MlpGateUp,
                members: gate_up,
                layer: Some(layer),
                expert: None,
                metadata: GroupMetadata {
                    name: format!("layer.{layer}.mlp_gate_up"),
                },
            });
        }
    }

    for raw in &metadata.tensors {
        if (raw.name.contains("mxfp4") || raw.name.contains("MXFP4"))
            && let Some(tensor) = graph.tensors.iter().find(|tensor| tensor.raw == raw.id)
        {
            graph.groups.push(SemanticGroup {
                kind: SemanticGroupKind::GptOssMxfp4,
                members: vec![tensor.id],
                layer: tensor.layer,
                expert: tensor.expert,
                metadata: GroupMetadata {
                    name: raw.name.clone(),
                },
            });
        }
    }
}

fn members_for_roles(graph: &SemanticGraph, layer: u32, roles: &[SemanticRole]) -> Vec<SemanticId> {
    roles
        .iter()
        .filter_map(|role| {
            graph
                .tensors_by_role(role, Some(layer))
                .next()
                .map(|tensor| tensor.id)
        })
        .collect()
}
