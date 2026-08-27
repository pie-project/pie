use model_dsl::Weight;
use model_loader::contract::{Expr, ModelContract, TensorContract, TensorType};

use super::model::{Mix, Mlp, Model};
use crate::contract::{ModelError, copy, declare, fused};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        if src.get("model.embed_tokens.weight").is_some() {
            return self.import_from_huggingface(src);
        }
        if src.get("token_embd.weight").is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Missing("model.embed_tokens.weight".to_string()))
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let tp = self.tp;
        let mut tensors = vec![
            copy(src, &self.embed, tp, "model.embed_tokens.weight")?,
            copy(src, &self.final_norm, tp, "model.norm.weight")?,
        ];

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("model.layers.{l}.{s}");
            let at = &w.attn;

            tensors.extend(gates(src, &w.attn_mix, tp, &n("hc_attn"))?);
            tensors.extend(gates(src, &w.mlp_mix, tp, &n("hc_mlp"))?);

            tensors.push(copy(src, &at.q_down, tp, n("self_attn.q_a_proj.weight"))?);
            tensors.push(copy(
                src,
                &at.q_norm,
                tp,
                n("self_attn.q_a_layernorm.weight"),
            )?);
            tensors.push(copy(src, &at.q_up, tp, n("self_attn.q_b_proj.weight"))?);
            tensors.push(copy(
                src,
                &at.kv_down,
                tp,
                n("self_attn.kv_a_proj_with_mqa.weight"),
            )?);
            tensors.push(copy(
                src,
                &at.kv_norm,
                tp,
                n("self_attn.kv_a_layernorm.weight"),
            )?);
            tensors.push(copy(src, &at.o_down, tp, n("self_attn.o_a_proj.weight"))?);
            tensors.push(copy(src, &at.o_up, tp, n("self_attn.o_b_proj.weight"))?);

            tensors.push(copy(src, &at.sink, tp, n("self_attn.sinks"))?);

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
                    bias,
                    gate_up,
                    down,
                    experts,
                    ..
                } => {
                    tensors.push(copy(src, router, tp, n("mlp.gate.weight"))?);

                    tensors.push(copy(src, bias, tp, n("mlp.gate.e_score_correction_bias"))?);

                    let inter = gate_up.dim(1) / 2;
                    let hidden = gate_up.dim(2);
                    let pair = |e: u32| {
                        let leg = |half: &str| {
                            lift(
                                gate_up,
                                n(&format!("mlp.experts.{e}.{half}.weight")),
                                [inter, hidden],
                            )
                        };
                        Expr::concat(1, vec![leg("gate_proj"), leg("up_proj")])
                    };
                    tensors.push(declare(
                        src,
                        gate_up,
                        tp,
                        Expr::concat(0, (0..*experts).map(pair).collect()),
                    )?);

                    let slab = |e: u32| {
                        lift(
                            down,
                            n(&format!("mlp.experts.{e}.down_proj.weight")),
                            [down.dim(1), down.dim(2)],
                        )
                    };
                    tensors.push(declare(
                        src,
                        down,
                        tp,
                        Expr::concat(0, (0..*experts).map(slab).collect()),
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

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");
            let at = &w.attn;

            tensors.extend(hyper(src, &w.attn_mix, tp, &n("hc_attn"))?);
            tensors.extend(hyper(src, &w.mlp_mix, tp, &n("hc_mlp"))?);

            tensors.push(copy(src, &at.q_down, tp, n("attn_q_a.weight"))?);
            tensors.push(copy(src, &at.q_norm, tp, n("attn_q_a_norm.weight"))?);
            tensors.push(copy(src, &at.q_up, tp, n("attn_q_b.weight"))?);
            tensors.push(copy(src, &at.kv_down, tp, n("attn_kv_a_mqa.weight"))?);
            tensors.push(copy(src, &at.kv_norm, tp, n("attn_kv_a_norm.weight"))?);
            tensors.push(copy(src, &at.o_down, tp, n("attn_o_a.weight"))?);
            tensors.push(copy(src, &at.o_up, tp, n("attn_o_b.weight"))?);

            tensors.push(copy(src, &at.sink, tp, n("attn_sinks"))?);

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
                    bias,
                    gate_up,
                    down,
                    ..
                } => {
                    tensors.push(copy(src, router, tp, n("ffn_gate_inp.weight"))?);

                    tensors.push(copy(src, bias, tp, n("exp_probs_b.bias"))?);

                    tensors.push(fused(
                        src,
                        gate_up,
                        tp,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?);

                    tensors.push(copy(src, down, tp, n("ffn_down_exps.weight"))?);
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

fn hyper(
    src: &ztensor::Source,
    mix: &Mix,
    tp: u32,
    from: &str,
) -> Result<[TensorContract; 2], ModelError> {
    Ok([
        copy(src, &mix.scale, tp, format!("{from}_scale.weight"))?,
        copy(src, &mix.base, tp, format!("{from}_base.weight"))?,
    ])
}

fn gates(
    src: &ztensor::Source,
    mix: &Mix,
    tp: u32,
    from: &str,
) -> Result<[TensorContract; 2], ModelError> {
    Ok([
        copy(src, &mix.scale, tp, format!("{from}_scale"))?,
        copy(src, &mix.base, tp, format!("{from}_base"))?,
    ])
}

fn lift(bank: &Weight, from: String, slab: [u64; 2]) -> Expr {
    Expr::src(from).transmute(TensorType::new(
        extents(&[1, slab[0], slab[1]]),
        crate::encoding(bank.dtype),
    ))
}

fn extents(shape: &[u64]) -> Vec<i64> {
    shape
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect()
}
