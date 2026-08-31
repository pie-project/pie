use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint::types::Encoding;

use super::model::{Mlp, Model};
use checkpoint_dsl::{Builder, Error, encoding};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        let huggingface = "model.embed_tokens.weight";
        if src.get(huggingface).is_some() {
            return self.import_from_huggingface(src);
        }
        Err(Error::Illegible {
            name: "glm5".to_string(),
            detail: format!("no `{huggingface}`: the one layout this family reads is huggingface"),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, Error> {
        let hidden = i64::from(self.hidden);
        let mut b = Builder::new(src, self.tp);
        b.read(&self.embed, "model.embed_tokens.weight")?;
        b.read(&self.final_norm, "model.norm.weight")?;
        b.read(&self.head, "lm_head.weight")?;
        for (l, layer) in self.layers.iter().enumerate() {
            let at = |tail: &str| format!("model.layers.{l}.{tail}");
            let attn = &layer.attn;
            let index = &attn.indexer;
            b.read(&layer.attn_norm, at("input_layernorm.weight"))?;
            b.read(&layer.mlp_norm, at("post_attention_layernorm.weight"))?;
            b.read(&attn.q_a_proj, at("self_attn.q_a_proj.weight"))?;
            b.read(&attn.q_a_norm, at("self_attn.q_a_layernorm.weight"))?;
            b.read(&attn.q_b_proj, at("self_attn.q_b_proj.weight"))?;
            b.read(&attn.kv_a_proj, at("self_attn.kv_a_proj_with_mqa.weight"))?;
            b.read(&attn.kv_a_norm, at("self_attn.kv_a_layernorm.weight"))?;
            b.read(&attn.kv_b_proj, at("self_attn.kv_b_proj.weight"))?;
            b.read(&attn.o_proj, at("self_attn.o_proj.weight"))?;
            b.read(&index.q_proj, at("self_attn.indexer.wq_b.weight"))?;
            b.read(&index.k_proj, at("self_attn.indexer.wk.weight"))?;
            b.read(&index.weights_proj, at("self_attn.indexer.weights_proj.weight"))?;
            b.read(&index.k_norm, at("self_attn.indexer.k_norm.weight"))?;
            b.read(&index.k_norm_bias, at("self_attn.indexer.k_norm.bias"))?;
            match &layer.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(
                        gate_up,
                        [at("mlp.gate_proj.weight"), at("mlp.up_proj.weight")],
                    )?;
                    b.read(down, at("mlp.down_proj.weight"))?;
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    experts,
                    ..
                } => {
                    b.read(router, at("mlp.gate.weight"))?;

                    b.read_expr(
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
                                        encoding(gate_up.dtype),
                                    )
                                })
                                .collect(),
                        ),
                    )?;

                    b.read_expr(
                        down,
                        Expr::concat(
                            0,
                            (0..*experts)
                                .map(|e| {
                                    slab(
                                        Expr::src(at(&format!("mlp.experts.{e}.down_proj.weight"))),
                                        vec![1, hidden, -1],
                                        encoding(down.dtype),
                                    )
                                })
                                .collect(),
                        ),
                    )?;
                    if let Some(shared) = shared {
                        b.read_concat(
                            &shared.gate_up,
                            [
                                at("mlp.shared_experts.gate_proj.weight"),
                                at("mlp.shared_experts.up_proj.weight"),
                            ],
                        )?;
                        b.read(&shared.down, at("mlp.shared_experts.down_proj.weight"))?;
                    }
                }
            }
        }
        Ok(b.build())
    }
}

fn slab(expr: Expr, shape: Vec<i64>, encoding: Encoding) -> Expr {
    expr.transmute(TensorType::new(shape, encoding))
}
