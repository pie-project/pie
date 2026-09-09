use checkpoint::contract::ModelContract;
use checkpoint_dsl::{Builder, Error};
use model_dsl::Platform;

use super::model::Model;

const TRUNK: &str = "model.language_model.";

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from_huggingface(src, platform)
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let at = |leaf: &str| format!("{TRUNK}{leaf}");
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, at("embed_tokens.weight"))?;
        b.read(&self.final_norm, at("norm.weight"))?;
        b.read(&self.lm_head, "lm_head.weight")?;

        for (l, w) in self.layers.iter().enumerate() {
            let n = |leaf: &str| format!("{TRUNK}layers.{l}.{leaf}");

            b.read(&w.attn_norm, n("input_layernorm.weight"))?;
            b.read(&w.post_attn_norm, n("post_attention_layernorm.weight"))?;
            b.read(&w.pre_ffw_norm, n("pre_feedforward_layernorm.weight"))?;
            b.read(&w.post_ffw_norm, n("post_feedforward_layernorm.weight"))?;

            b.read_concat(
                &w.qkv,
                [
                    n("self_attn.q_proj.weight"),
                    n("self_attn.k_proj.weight"),
                    n("self_attn.v_proj.weight"),
                ],
            )?;
            b.read(&w.gate, n("self_attn.gate_proj.weight"))?;
            b.read(&w.o_proj, n("self_attn.o_proj.weight"))?;

            b.read_concat(
                &w.gate_up,
                [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
            )?;
            b.read(&w.down, n("mlp.down_proj.weight"))?;
        }

        Ok(b.build())
    }
}
