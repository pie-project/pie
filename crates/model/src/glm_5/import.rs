use model_loader::contract::{Expr, ModelContract, TensorType};
use model_loader::types::Encoding;

use super::model::{Mlp, Model};
use crate::contract::{ALIGNMENT, ModelError, copy, declare, fused};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        let huggingface = "model.embed_tokens.weight";
        if src.get(huggingface).is_some() {
            return self.import_from_huggingface(src);
        }
        Err(ModelError::Illegible {
            name: "glm5".to_string(),
            detail: format!("no `{huggingface}`: the one layout this family reads is huggingface"),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let hidden = i64::from(self.hidden);
        let mut tensors = Vec::new();
        tensors.push(copy(src, &self.embed, "model.embed_tokens.weight")?);
        tensors.push(copy(src, &self.final_norm, "model.norm.weight")?);
        tensors.push(copy(src, &self.head, "lm_head.weight")?);
        for (l, layer) in self.layers.iter().enumerate() {
            let at = |tail: &str| format!("model.layers.{l}.{tail}");
            let attn = &layer.attn;
            let index = &attn.indexer;
            tensors.push(copy(src, &layer.attn_norm, at("input_layernorm.weight"))?);
            tensors.push(copy(
                src,
                &layer.mlp_norm,
                at("post_attention_layernorm.weight"),
            )?);
            tensors.push(copy(src, &attn.q_a_proj, at("self_attn.q_a_proj.weight"))?);
            tensors.push(copy(
                src,
                &attn.q_a_norm,
                at("self_attn.q_a_layernorm.weight"),
            )?);
            tensors.push(copy(src, &attn.q_b_proj, at("self_attn.q_b_proj.weight"))?);
            tensors.push(copy(
                src,
                &attn.kv_a_proj,
                at("self_attn.kv_a_proj_with_mqa.weight"),
            )?);
            tensors.push(copy(
                src,
                &attn.kv_a_norm,
                at("self_attn.kv_a_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &attn.kv_b_proj,
                at("self_attn.kv_b_proj.weight"),
            )?);
            tensors.push(copy(src, &attn.o_proj, at("self_attn.o_proj.weight"))?);
            tensors.push(copy(
                src,
                &index.q_proj,
                at("self_attn.indexer.wq_b.weight"),
            )?);
            tensors.push(copy(src, &index.k_proj, at("self_attn.indexer.wk.weight"))?);
            tensors.push(copy(
                src,
                &index.weights_proj,
                at("self_attn.indexer.weights_proj.weight"),
            )?);
            tensors.push(copy(
                src,
                &index.k_norm,
                at("self_attn.indexer.k_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &index.k_norm_bias,
                at("self_attn.indexer.k_norm.bias"),
            )?);
            match &layer.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [at("mlp.gate_proj.weight"), at("mlp.up_proj.weight")],
                    )?);
                    tensors.push(copy(src, down, at("mlp.down_proj.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    experts,
                    ..
                } => {
                    tensors.push(copy(src, router, at("mlp.gate.weight"))?);

                    tensors.push(declare(
                        src,
                        gate_up,
                        Expr::concat(
                            0,
                            (0..*experts)
                                .map(|e| {
                                    slab(
                                        Expr::concat(
                                            0,
                                            vec![
                                                Expr::src(at(&format!(
                                                    "mlp.experts.{e}.gate_proj.weight"
                                                ))),
                                                Expr::src(at(&format!(
                                                    "mlp.experts.{e}.up_proj.weight"
                                                ))),
                                            ],
                                        ),
                                        vec![1, -1, hidden],
                                        crate::encoding(gate_up.dtype),
                                    )
                                })
                                .collect(),
                        ),
                    )?);

                    tensors.push(declare(
                        src,
                        down,
                        Expr::concat(
                            0,
                            (0..*experts)
                                .map(|e| {
                                    slab(
                                        Expr::src(at(&format!("mlp.experts.{e}.down_proj.weight"))),
                                        vec![1, hidden, -1],
                                        crate::encoding(down.dtype),
                                    )
                                })
                                .collect(),
                        ),
                    )?);
                    if let Some(shared) = shared {
                        tensors.push(fused(
                            src,
                            &shared.gate_up,
                            [
                                at("mlp.shared_experts.gate_proj.weight"),
                                at("mlp.shared_experts.up_proj.weight"),
                            ],
                        )?);
                        tensors.push(copy(
                            src,
                            &shared.down,
                            at("mlp.shared_experts.down_proj.weight"),
                        )?);
                    }
                }
            }
        }
        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,
            groups: Vec::new(),
        })
    }
}

fn slab(expr: Expr, shape: Vec<i64>, encoding: Encoding) -> Expr {
    expr.transmute(TensorType::new(shape, encoding))
}
