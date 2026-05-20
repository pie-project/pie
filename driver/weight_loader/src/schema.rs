use crate::config::ModelConfig;
use crate::error::CompileError;
use crate::semantic::SemanticGraph;
use crate::source::CheckpointMetadata;

pub trait ModelSchema: Send + Sync {
    fn matches(&self, model_type: &str) -> bool;
    fn build(
        &self,
        metadata: &CheckpointMetadata,
        cfg: &ModelConfig,
    ) -> Result<SemanticGraph, CompileError>;
}

pub fn build_semantic_graph(
    metadata: &CheckpointMetadata,
    cfg: &ModelConfig,
) -> Result<SemanticGraph, CompileError> {
    let schema = find_schema(&cfg.model_type).unwrap_or(crate::schemas::generic_schema());
    schema.build(metadata, cfg)
}

pub fn find_schema(model_type: &str) -> Option<&'static dyn ModelSchema> {
    crate::schemas::builtin_schemas()
        .iter()
        .copied()
        .find(|schema| schema.matches(model_type))
}
