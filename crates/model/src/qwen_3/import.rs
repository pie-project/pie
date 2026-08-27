use model_loader::contract::{Expr, ModelContract, TensorType};

use super::model::{Head, Mixer, Mlp, Model};
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
            name: "qwen_3".to_string(),
            detail: format!(
                "it holds neither `{huggingface}` nor `{gguf}`, so it is written \
                 in neither format this family reads"
            ),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let mut tensors = vec![
            copy(src, &self.embed, "model.language_model.embed_tokens.weight")?,
            copy(src, &self.final_norm, "model.language_model.norm.weight")?,
        ];

        if let Head::Bank(head) = &self.head {
            tensors.push(copy(src, head, "lm_head.weight")?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("model.language_model.layers.{l}.{s}");

            tensors.push(copy(src, &w.mixer_norm, n("input_layernorm.weight"))?);
            tensors.push(copy(
                src,
                &w.mlp_norm,
                n("post_attention_layernorm.weight"),
            )?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.push(copy(src, &a.qg_proj, n("self_attn.q_proj.weight"))?);
                    tensors.push(copy(src, &a.k_proj, n("self_attn.k_proj.weight"))?);
                    tensors.push(copy(src, &a.v_proj, n("self_attn.v_proj.weight"))?);
                    tensors.push(copy(src, &a.o_proj, n("self_attn.o_proj.weight"))?);
                    tensors.push(copy(src, &a.q_norm, n("self_attn.q_norm.weight"))?);
                    tensors.push(copy(src, &a.k_norm, n("self_attn.k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.push(fused(
                        src,
                        &g.in_qkvz,
                        [
                            n("linear_attn.in_proj_qkv.weight"),
                            n("linear_attn.in_proj_z.weight"),
                        ],
                    )?);

                    tensors.push(fused(
                        src,
                        &g.in_ba,
                        [
                            n("linear_attn.in_proj_b.weight"),
                            n("linear_attn.in_proj_a.weight"),
                        ],
                    )?);

                    tensors.push(declare(
                        src,
                        &g.conv,
                        squeezed(src, n("linear_attn.conv1d.weight"))?,
                    )?);

                    tensors.push(copy(src, &g.dt_bias, n("linear_attn.dt_bias"))?);
                    tensors.push(copy(src, &g.a_log, n("linear_attn.A_log"))?);
                    tensors.push(copy(src, &g.norm, n("linear_attn.norm.weight"))?);
                    tensors.push(copy(src, &g.out_proj, n("linear_attn.out_proj.weight"))?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?);
                    tensors.push(copy(src, down, n("mlp.down_proj.weight"))?);
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
                    tensors.push(copy(src, router, n("mlp.gate.weight"))?);

                    tensors.push(copy(src, gate_up, n("mlp.experts.gate_up_proj"))?);
                    tensors.push(copy(src, down, n("mlp.experts.down_proj"))?);
                    tensors.push(fused(
                        src,
                        shared_gate_up,
                        [
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    )?);
                    tensors.push(copy(
                        src,
                        shared_down,
                        n("mlp.shared_expert.down_proj.weight"),
                    )?);
                    tensors.push(copy(src, shared_gate, n("mlp.shared_expert_gate.weight"))?);
                }
            }
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let mut tensors = vec![
            copy(src, &self.embed, "token_embd.weight")?,
            copy(src, &self.final_norm, "output_norm.weight")?,
        ];

        if let Head::Bank(head) = &self.head {
            tensors.push(copy(src, head, "output.weight")?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");

            tensors.push(copy(src, &w.mixer_norm, n("attn_norm.weight"))?);
            tensors.push(copy(src, &w.mlp_norm, n("ffn_norm.weight"))?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.push(copy(src, &a.qg_proj, n("attn_q.weight"))?);
                    tensors.push(copy(src, &a.k_proj, n("attn_k.weight"))?);
                    tensors.push(copy(src, &a.v_proj, n("attn_v.weight"))?);
                    tensors.push(copy(src, &a.o_proj, n("attn_output.weight"))?);
                    tensors.push(copy(src, &a.q_norm, n("attn_q_norm.weight"))?);
                    tensors.push(copy(src, &a.k_norm, n("attn_k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.push(copy(src, &g.in_qkvz, n("ssm_in.weight"))?);
                    tensors.push(copy(src, &g.in_ba, n("ssm_beta_alpha.weight"))?);
                    tensors.push(copy(src, &g.conv, n("ssm_conv1d.weight"))?);
                    tensors.push(copy(src, &g.dt_bias, n("ssm_dt.bias"))?);
                    tensors.push(copy(src, &g.a_log, n("ssm_a"))?);
                    tensors.push(copy(src, &g.norm, n("ssm_norm.weight"))?);
                    tensors.push(copy(src, &g.out_proj, n("ssm_out.weight"))?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("ffn_gate.weight"), n("ffn_up.weight")],
                    )?);
                    tensors.push(copy(src, down, n("ffn_down.weight"))?);
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
                    tensors.push(copy(src, router, n("ffn_gate_inp.weight"))?);

                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?);
                    tensors.push(copy(src, down, n("ffn_down_exps.weight"))?);
                    tensors.push(fused(
                        src,
                        shared_gate_up,
                        [n("ffn_gate_shexp.weight"), n("ffn_up_shexp.weight")],
                    )?);
                    tensors.push(copy(src, shared_down, n("ffn_down_shexp.weight"))?);
                    tensors.push(copy(src, shared_gate, n("ffn_gate_inp_shexp.weight"))?);
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

fn squeezed(src: &ztensor::Source, from: String) -> Result<Expr, ModelError> {
    let Some(tensor) = src.get(&from) else {
        return Err(ModelError::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| ModelError::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let [channels, 1, kernel] = *shape else {
        return Err(illegible(&format!(
            "a depthwise convolution bank is stored [channels, 1, kernel] and \
             this one is stored {shape:?}"
        )));
    };
    let part = tensor.part("data").map_err(|why| illegible(&why))?;
    let stored =
        model_loader::checkpoint::encoding_of(&tensor, &part).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(
        vec![extent(channels), extent(kernel)],
        stored,
    )))
}

fn extent(of: u64) -> i64 {
    i64::try_from(of).expect("an extent no i64 holds")
}
