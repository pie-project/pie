use model_loader::contract::{Expr, ModelContract};

use super::model::{AttnBanks, Model};
use crate::contract::{ModelError, copy, declare, fused};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let embed = "model.language_model.embed_tokens.weight";
        if src.get(embed).is_some() {
            return self.import_from_huggingface(src);
        }
        if src.get("token_embd.weight").is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Missing(embed.to_string()))
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let tp = self.tp;
        let mut tensors = Vec::new();

        tensors.push(copy(
            src,
            &self.embed,
            tp,
            "model.language_model.embed_tokens.weight",
        )?);
        tensors.push(copy(
            src,
            &self.final_norm,
            tp,
            "model.language_model.norm.weight",
        )?);

        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(
                src,
                &w.attn_norm,
                tp,
                format!("model.language_model.layers.{l}.input_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_attn_norm,
                tp,
                format!("model.language_model.layers.{l}.post_attention_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.pre_ffw_norm,
                tp,
                format!("model.language_model.layers.{l}.pre_feedforward_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_ffw_norm,
                tp,
                format!("model.language_model.layers.{l}.post_feedforward_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.attn.q_norm,
                tp,
                format!("model.language_model.layers.{l}.self_attn.q_norm.weight"),
            )?);
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    tensors.push(copy(
                        src,
                        k_norm,
                        tp,
                        format!("model.language_model.layers.{l}.self_attn.k_norm.weight"),
                    )?);
                    tensors.push(fused(
                        src,
                        qkv,
                        tp,
                        [
                            format!("model.language_model.layers.{l}.self_attn.q_proj.weight"),
                            format!("model.language_model.layers.{l}.self_attn.k_proj.weight"),
                            format!("model.language_model.layers.{l}.self_attn.v_proj.weight"),
                        ],
                    )?);
                }

                AttnBanks::Shared { q_proj } => {
                    tensors.push(copy(
                        src,
                        q_proj,
                        tp,
                        format!("model.language_model.layers.{l}.self_attn.q_proj.weight"),
                    )?);
                }
            }
            tensors.push(copy(
                src,
                &w.o_proj,
                tp,
                format!("model.language_model.layers.{l}.self_attn.o_proj.weight"),
            )?);
            tensors.push(fused(
                src,
                &w.gate_up,
                tp,
                [
                    format!("model.language_model.layers.{l}.mlp.gate_proj.weight"),
                    format!("model.language_model.layers.{l}.mlp.up_proj.weight"),
                ],
            )?);
            tensors.push(copy(
                src,
                &w.down,
                tp,
                format!("model.language_model.layers.{l}.mlp.down_proj.weight"),
            )?);
        }

        if let Some(ple) = &self.ple {
            tensors.push(copy(
                src,
                &ple.model_proj,
                tp,
                "model.language_model.per_layer_model_projection.weight",
            )?);
            tensors.push(copy(
                src,
                &ple.model_norm,
                tp,
                "model.language_model.per_layer_projection_norm.weight",
            )?);
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                tensors.push(declare(
                    src,
                    &p.table,
                    tp,
                    Expr::src("model.language_model.embed_tokens_per_layer.weight")
                        .slice(1, at, width),
                )?);
                tensors.push(copy(
                    src,
                    &p.gate,
                    tp,
                    format!("model.language_model.layers.{l}.per_layer_input_gate.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &p.proj,
                    tp,
                    format!("model.language_model.layers.{l}.per_layer_projection.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &p.norm,
                    tp,
                    format!("model.language_model.layers.{l}.post_per_layer_input_norm.weight"),
                )?);

                tensors.push(copy(
                    src,
                    &p.scalar,
                    tp,
                    format!("model.language_model.layers.{l}.layer_scalar"),
                )?);
            }
        }

        Ok(ModelContract {
            alignment: 256,
            tensors,

            groups: Vec::new(),
        })
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let tp = self.tp;
        let mut tensors = Vec::new();

        tensors.push(copy(src, &self.embed, tp, "token_embd.weight")?);
        tensors.push(copy(src, &self.final_norm, tp, "output_norm.weight")?);

        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(
                src,
                &w.attn_norm,
                tp,
                format!("blk.{l}.attn_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_attn_norm,
                tp,
                format!("blk.{l}.post_attention_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.pre_ffw_norm,
                tp,
                format!("blk.{l}.ffn_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_ffw_norm,
                tp,
                format!("blk.{l}.post_ffw_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.attn.q_norm,
                tp,
                format!("blk.{l}.attn_q_norm.weight"),
            )?);
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    tensors.push(copy(
                        src,
                        k_norm,
                        tp,
                        format!("blk.{l}.attn_k_norm.weight"),
                    )?);
                    tensors.push(fused(
                        src,
                        qkv,
                        tp,
                        [
                            format!("blk.{l}.attn_q.weight"),
                            format!("blk.{l}.attn_k.weight"),
                            format!("blk.{l}.attn_v.weight"),
                        ],
                    )?);
                }

                AttnBanks::Shared { q_proj } => {
                    tensors.push(copy(src, q_proj, tp, format!("blk.{l}.attn_q.weight"))?);
                }
            }
            tensors.push(copy(
                src,
                &w.o_proj,
                tp,
                format!("blk.{l}.attn_output.weight"),
            )?);
            tensors.push(fused(
                src,
                &w.gate_up,
                tp,
                [
                    format!("blk.{l}.ffn_gate.weight"),
                    format!("blk.{l}.ffn_up.weight"),
                ],
            )?);
            tensors.push(copy(src, &w.down, tp, format!("blk.{l}.ffn_down.weight"))?);
        }

        if let Some(ple) = &self.ple {
            tensors.push(copy(
                src,
                &ple.model_proj,
                tp,
                "per_layer_model_proj.weight",
            )?);
            tensors.push(copy(
                src,
                &ple.model_norm,
                tp,
                "per_layer_proj_norm.weight",
            )?);
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                tensors.push(declare(
                    src,
                    &p.table,
                    tp,
                    Expr::src("per_layer_token_embd.weight").slice(1, at, width),
                )?);
                tensors.push(copy(src, &p.gate, tp, format!("blk.{l}.inp_gate.weight"))?);
                tensors.push(copy(src, &p.proj, tp, format!("blk.{l}.proj.weight"))?);
                tensors.push(copy(src, &p.norm, tp, format!("blk.{l}.post_norm.weight"))?);

                tensors.push(copy(src, &p.scalar, tp, format!("blk.{l}.layer_scalar"))?);
            }
        }

        Ok(ModelContract {
            alignment: 256,
            tensors,

            groups: Vec::new(),
        })
    }
}
