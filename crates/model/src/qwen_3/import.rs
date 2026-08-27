use model_dsl::Weight;
use model_loader::contract::{Expr, ModelContract, TensorType};

use super::model::{Head, Mixer, Mlp, Model};
use crate::contract::{ModelError, copy, declare, fused};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let huggingface = "model.language_model.embed_tokens.weight";
        if src.get(huggingface).is_some() {
            return self.import_from_huggingface(src);
        }
        if src.get("token_embd.weight").is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Missing(huggingface.to_string()))
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let tp = self.tp;
        let mut tensors = vec![
            copy(
                src,
                &self.embed,
                tp,
                "model.language_model.embed_tokens.weight",
            )?,
            copy(
                src,
                &self.final_norm,
                tp,
                "model.language_model.norm.weight",
            )?,
        ];

        if let Head::Bank(head) = &self.head {
            tensors.push(copy(src, head, tp, "lm_head.weight")?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("model.language_model.layers.{l}.{s}");

            tensors.push(copy(src, &w.mixer_norm, tp, n("input_layernorm.weight"))?);
            tensors.push(copy(
                src,
                &w.mlp_norm,
                tp,
                n("post_attention_layernorm.weight"),
            )?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.push(copy(src, &a.qg_proj, tp, n("self_attn.q_proj.weight"))?);
                    tensors.push(copy(src, &a.k_proj, tp, n("self_attn.k_proj.weight"))?);
                    tensors.push(copy(src, &a.v_proj, tp, n("self_attn.v_proj.weight"))?);
                    tensors.push(copy(src, &a.o_proj, tp, n("self_attn.o_proj.weight"))?);
                    tensors.push(copy(src, &a.q_norm, tp, n("self_attn.q_norm.weight"))?);
                    tensors.push(copy(src, &a.k_norm, tp, n("self_attn.k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.push(fused(
                        src,
                        &g.in_qkvz,
                        tp,
                        [
                            n("linear_attn.in_proj_qkv.weight"),
                            n("linear_attn.in_proj_z.weight"),
                        ],
                    )?);

                    tensors.push(fused(
                        src,
                        &g.in_ba,
                        tp,
                        [
                            n("linear_attn.in_proj_b.weight"),
                            n("linear_attn.in_proj_a.weight"),
                        ],
                    )?);

                    tensors.push(declare(
                        src,
                        &g.conv,
                        tp,
                        squeezed(&g.conv, n("linear_attn.conv1d.weight")),
                    )?);

                    tensors.push(copy(src, &g.dt_bias, tp, n("linear_attn.dt_bias"))?);
                    tensors.push(copy(src, &g.a_log, tp, n("linear_attn.A_log"))?);
                    tensors.push(copy(src, &g.norm, tp, n("linear_attn.norm.weight"))?);
                    tensors.push(copy(
                        src,
                        &g.out_proj,
                        tp,
                        n("linear_attn.out_proj.weight"),
                    )?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        tp,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?);
                    tensors.push(copy(src, down, tp, n("mlp.down_proj.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    ..
                } => {
                    tensors.push(copy(src, router, tp, n("mlp.gate.weight"))?);

                    tensors.push(copy(src, gate_up, tp, n("mlp.experts.gate_up_proj"))?);
                    tensors.push(copy(src, down, tp, n("mlp.experts.down_proj"))?);
                    tensors.push(fused(
                        src,
                        shared_gate_up,
                        tp,
                        [
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    )?);
                    tensors.push(copy(
                        src,
                        shared_down,
                        tp,
                        n("mlp.shared_expert.down_proj.weight"),
                    )?);
                    tensors.push(copy(
                        src,
                        shared_gate,
                        tp,
                        n("mlp.shared_expert_gate.weight"),
                    )?);
                }
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
        let mut tensors = vec![
            copy(src, &self.embed, tp, "token_embd.weight")?,
            copy(src, &self.final_norm, tp, "output_norm.weight")?,
        ];

        if let Head::Bank(head) = &self.head {
            tensors.push(copy(src, head, tp, "output.weight")?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");

            tensors.push(copy(src, &w.mixer_norm, tp, n("attn_norm.weight"))?);
            tensors.push(copy(src, &w.mlp_norm, tp, n("ffn_norm.weight"))?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.push(copy(src, &a.qg_proj, tp, n("attn_q.weight"))?);
                    tensors.push(copy(src, &a.k_proj, tp, n("attn_k.weight"))?);
                    tensors.push(copy(src, &a.v_proj, tp, n("attn_v.weight"))?);
                    tensors.push(copy(src, &a.o_proj, tp, n("attn_output.weight"))?);
                    tensors.push(copy(src, &a.q_norm, tp, n("attn_q_norm.weight"))?);
                    tensors.push(copy(src, &a.k_norm, tp, n("attn_k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.push(copy(src, &g.in_qkvz, tp, n("ssm_in.weight"))?);
                    tensors.push(copy(src, &g.in_ba, tp, n("ssm_beta_alpha.weight"))?);
                    tensors.push(copy(src, &g.conv, tp, n("ssm_conv1d.weight"))?);
                    tensors.push(copy(src, &g.dt_bias, tp, n("ssm_dt.bias"))?);
                    tensors.push(copy(src, &g.a_log, tp, n("ssm_a"))?);
                    tensors.push(copy(src, &g.norm, tp, n("ssm_norm.weight"))?);
                    tensors.push(copy(src, &g.out_proj, tp, n("ssm_out.weight"))?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        tp,
                        [n("ffn_gate.weight"), n("ffn_up.weight")],
                    )?);
                    tensors.push(copy(src, down, tp, n("ffn_down.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    ..
                } => {
                    tensors.push(copy(src, router, tp, n("ffn_gate_inp.weight"))?);

                    tensors.push(fused(
                        src,
                        gate_up,
                        tp,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?);
                    tensors.push(copy(src, down, tp, n("ffn_down_exps.weight"))?);
                    tensors.push(fused(
                        src,
                        shared_gate_up,
                        tp,
                        [n("ffn_gate_shexp.weight"), n("ffn_up_shexp.weight")],
                    )?);
                    tensors.push(copy(src, shared_down, tp, n("ffn_down_shexp.weight"))?);
                    tensors.push(copy(src, shared_gate, tp, n("ffn_gate_inp_shexp.weight"))?);
                }
            }
        }

        Ok(ModelContract {
            alignment: 256,
            tensors,

            groups: Vec::new(),
        })
    }
}

fn squeezed(w: &Weight, from: String) -> Expr {
    Expr::src(from).transmute(TensorType::new(extents(&w.shape), crate::encoding(w.dtype)))
}

fn extents(shape: &[u64]) -> Vec<i64> {
    shape
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect()
}
