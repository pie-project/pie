use model_loader::contract::{Expr, ModelContract};

use super::model::{AttnBanks, Model};
use crate::contract::{ALIGNMENT, ModelError, copy, declare, fused};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        let huggingface = "model.language_model.embed_tokens.weight";
        let gguf = "token_embd.weight";
        if src.get(huggingface).is_some() {
            return self.import_from_huggingface(src);
        }
        if src.get(gguf).is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Illegible {
            name: "gemma4".to_string(),
            detail: format!("it holds neither `{huggingface}` nor `{gguf}`"),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let mut tensors = Vec::new();

        tensors.push(copy(
            src,
            &self.embed,
            "model.language_model.embed_tokens.weight",
        )?);
        tensors.push(copy(
            src,
            &self.final_norm,
            "model.language_model.norm.weight",
        )?);

        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(
                src,
                &w.attn_norm,
                format!("model.language_model.layers.{l}.input_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_attn_norm,
                format!("model.language_model.layers.{l}.post_attention_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.pre_ffw_norm,
                format!("model.language_model.layers.{l}.pre_feedforward_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_ffw_norm,
                format!("model.language_model.layers.{l}.post_feedforward_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.attn.q_norm,
                format!("model.language_model.layers.{l}.self_attn.q_norm.weight"),
            )?);
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    tensors.push(copy(
                        src,
                        k_norm,
                        format!("model.language_model.layers.{l}.self_attn.k_norm.weight"),
                    )?);
                    tensors.push(fused(
                        src,
                        qkv,
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
                        format!("model.language_model.layers.{l}.self_attn.q_proj.weight"),
                    )?);
                }
            }
            tensors.push(copy(
                src,
                &w.o_proj,
                format!("model.language_model.layers.{l}.self_attn.o_proj.weight"),
            )?);
            tensors.push(fused(
                src,
                &w.gate_up,
                [
                    format!("model.language_model.layers.{l}.mlp.gate_proj.weight"),
                    format!("model.language_model.layers.{l}.mlp.up_proj.weight"),
                ],
            )?);
            tensors.push(copy(
                src,
                &w.down,
                format!("model.language_model.layers.{l}.mlp.down_proj.weight"),
            )?);
        }

        if let Some(ple) = &self.ple {
            tensors.push(copy(
                src,
                &ple.model_proj,
                "model.language_model.per_layer_model_projection.weight",
            )?);
            tensors.push(copy(
                src,
                &ple.model_norm,
                "model.language_model.per_layer_projection_norm.weight",
            )?);
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                tensors.push(declare(
                    src,
                    &p.table,
                    Expr::src("model.language_model.embed_tokens_per_layer.weight")
                        .slice(1, at, width),
                )?);
                tensors.push(copy(
                    src,
                    &p.gate,
                    format!("model.language_model.layers.{l}.per_layer_input_gate.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &p.proj,
                    format!("model.language_model.layers.{l}.per_layer_projection.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &p.norm,
                    format!("model.language_model.layers.{l}.post_per_layer_input_norm.weight"),
                )?);

                tensors.push(copy(
                    src,
                    &p.scalar,
                    format!("model.language_model.layers.{l}.layer_scalar"),
                )?);
            }
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let mut tensors = Vec::new();

        tensors.push(copy(src, &self.embed, "token_embd.weight")?);
        tensors.push(copy(src, &self.final_norm, "output_norm.weight")?);

        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(
                src,
                &w.attn_norm,
                format!("blk.{l}.attn_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_attn_norm,
                format!("blk.{l}.post_attention_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.pre_ffw_norm,
                format!("blk.{l}.ffn_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_ffw_norm,
                format!("blk.{l}.post_ffw_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.attn.q_norm,
                format!("blk.{l}.attn_q_norm.weight"),
            )?);
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    tensors.push(copy(src, k_norm, format!("blk.{l}.attn_k_norm.weight"))?);
                    tensors.push(fused(
                        src,
                        qkv,
                        [
                            format!("blk.{l}.attn_q.weight"),
                            format!("blk.{l}.attn_k.weight"),
                            format!("blk.{l}.attn_v.weight"),
                        ],
                    )?);
                }

                AttnBanks::Shared { q_proj } => {
                    tensors.push(copy(src, q_proj, format!("blk.{l}.attn_q.weight"))?);
                }
            }
            tensors.push(copy(src, &w.o_proj, format!("blk.{l}.attn_output.weight"))?);
            tensors.push(fused(
                src,
                &w.gate_up,
                [
                    format!("blk.{l}.ffn_gate.weight"),
                    format!("blk.{l}.ffn_up.weight"),
                ],
            )?);
            tensors.push(copy(src, &w.down, format!("blk.{l}.ffn_down.weight"))?);
        }

        if let Some(ple) = &self.ple {
            tensors.push(copy(src, &ple.model_proj, "per_layer_model_proj.weight")?);
            tensors.push(copy(src, &ple.model_norm, "per_layer_proj_norm.weight")?);
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                tensors.push(declare(
                    src,
                    &p.table,
                    Expr::src("per_layer_token_embd.weight").slice(1, at, width),
                )?);
                tensors.push(copy(src, &p.gate, format!("blk.{l}.inp_gate.weight"))?);
                tensors.push(copy(src, &p.proj, format!("blk.{l}.proj.weight"))?);
                tensors.push(copy(src, &p.norm, format!("blk.{l}.post_norm.weight"))?);

                tensors.push(copy(src, &p.scalar, format!("blk.{l}.layer_scalar"))?);
            }
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }
}
