use model_dsl::Dtype;
use model_loader::contract::{Expr, ModelContract, TensorType};

use super::model::{Mlp, Model};
use crate::contract::{ALIGNMENT, ModelError, copy, declare, fused};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        if src.get("model.embed_tokens.weight").is_some() {
            return self.import_from_huggingface(src);
        }
        if src.get("token_embd.weight").is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Illegible {
            name: "dsv4".to_string(),
            detail: "neither `model.embed_tokens.weight` (huggingface) nor \
                     `token_embd.weight` (gguf) names a tensor here"
                .to_string(),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let mut tensors = vec![
            copy(src, &self.embed, "model.embed_tokens.weight")?,
            copy(src, &self.final_norm, "model.norm.weight")?,
        ];

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("model.layers.{l}.{s}");
            let at = &w.attn;

            tensors.push(copy(src, &w.attn_mix.scale, n("hc_attn_scale"))?);
            tensors.push(copy(src, &w.attn_mix.base, n("hc_attn_base"))?);
            tensors.push(copy(src, &w.mlp_mix.scale, n("hc_mlp_scale"))?);
            tensors.push(copy(src, &w.mlp_mix.base, n("hc_mlp_base"))?);

            tensors.push(copy(src, &at.q_down, n("self_attn.q_a_proj.weight"))?);
            tensors.push(copy(src, &at.q_norm, n("self_attn.q_a_layernorm.weight"))?);
            tensors.push(copy(src, &at.q_up, n("self_attn.q_b_proj.weight"))?);
            tensors.push(copy(
                src,
                &at.kv_down,
                n("self_attn.kv_a_proj_with_mqa.weight"),
            )?);
            tensors.push(copy(
                src,
                &at.kv_norm,
                n("self_attn.kv_a_layernorm.weight"),
            )?);
            tensors.push(copy(src, &at.o_down, n("self_attn.o_a_proj.weight"))?);
            tensors.push(copy(src, &at.o_up, n("self_attn.o_b_proj.weight"))?);

            tensors.push(copy(src, &at.sink, n("self_attn.sinks"))?);

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
                    bias,
                    gate_up,
                    down,
                    experts,
                    ..
                } => {
                    tensors.push(copy(src, router, n("mlp.gate.weight"))?);

                    tensors.push(copy(src, bias, n("mlp.gate.e_score_correction_bias"))?);

                    let inter = gate_up.dim(1) / 2;
                    let hidden = gate_up.dim(2);
                    let pair = |e: u32| {
                        let leg = |half: &str| {
                            one_bank_row(
                                gate_up.dtype,
                                n(&format!("mlp.experts.{e}.{half}.weight")),
                                inter,
                                hidden,
                            )
                        };
                        Expr::concat(1, vec![leg("gate_proj"), leg("up_proj")])
                    };
                    tensors.push(declare(
                        src,
                        gate_up,
                        Expr::concat(0, (0..*experts).map(pair).collect()),
                    )?);

                    let slab = |e: u32| {
                        one_bank_row(
                            down.dtype,
                            n(&format!("mlp.experts.{e}.down_proj.weight")),
                            down.dim(1),
                            down.dim(2),
                        )
                    };
                    tensors.push(declare(
                        src,
                        down,
                        Expr::concat(0, (0..*experts).map(slab).collect()),
                    )?);
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

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");
            let at = &w.attn;

            tensors.push(copy(src, &w.attn_mix.scale, n("hc_attn_scale.weight"))?);
            tensors.push(copy(src, &w.attn_mix.base, n("hc_attn_base.weight"))?);
            tensors.push(copy(src, &w.mlp_mix.scale, n("hc_mlp_scale.weight"))?);
            tensors.push(copy(src, &w.mlp_mix.base, n("hc_mlp_base.weight"))?);

            tensors.push(copy(src, &at.q_down, n("attn_q_a.weight"))?);
            tensors.push(copy(src, &at.q_norm, n("attn_q_a_norm.weight"))?);
            tensors.push(copy(src, &at.q_up, n("attn_q_b.weight"))?);
            tensors.push(copy(src, &at.kv_down, n("attn_kv_a_mqa.weight"))?);
            tensors.push(copy(src, &at.kv_norm, n("attn_kv_a_norm.weight"))?);
            tensors.push(copy(src, &at.o_down, n("attn_o_a.weight"))?);
            tensors.push(copy(src, &at.o_up, n("attn_o_b.weight"))?);

            tensors.push(copy(src, &at.sink, n("attn_sinks"))?);

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
                    bias,
                    gate_up,
                    down,
                    ..
                } => {
                    tensors.push(copy(src, router, n("ffn_gate_inp.weight"))?);

                    tensors.push(copy(src, bias, n("exp_probs_b.bias"))?);

                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?);

                    tensors.push(copy(src, down, n("ffn_down_exps.weight"))?);
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

fn one_bank_row(dtype: Dtype, from: String, rows: u64, cols: u64) -> Expr {
    let extent = |e: u64| i64::try_from(e).expect("an extent no i64 holds");
    Expr::src(from).transmute(TensorType::new(
        vec![1, extent(rows), extent(cols)],
        crate::encoding(dtype),
    ))
}
