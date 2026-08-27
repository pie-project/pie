use model_dsl::{Shard, Weight};
use model_loader::contract::{Expr, ModelContract, TensorContract, TensorType};
use model_loader::types::DType;

use super::model::{Kda, Mixer, Mla, Mlp, Model};
use crate::contract::{ModelError, copy, declare, fused};

const HF_EMBED: &str = "language_model.model.embed_tokens.weight";

const GGUF_EMBED: &str = "token_embd.weight";

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        if src.get(HF_EMBED).is_some() {
            self.import_from_huggingface(src)
        } else if src.get(GGUF_EMBED).is_some() {
            self.import_from_gguf(src)
        } else {
            Err(ModelError::Missing(HF_EMBED.to_string()))
        }
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let mut tensors = Vec::new();
        tensors.push(copy(src, &self.embed, self.tp, HF_EMBED)?);
        tensors.push(copy(
            src,
            &self.final_norm,
            self.tp,
            "language_model.model.norm.weight",
        )?);
        tensors.push(copy(
            src,
            &self.head,
            self.tp,
            "language_model.lm_head.weight",
        )?);
        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(
                src,
                &w.mixer_norm,
                self.tp,
                at(l, "input_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.mlp_norm,
                self.tp,
                at(l, "post_attention_layernorm.weight"),
            )?);
            if let Some(res) = &w.res_blend {
                tensors.push(copy(
                    src,
                    &res.norm,
                    self.tp,
                    at(l, "self_attention_res_norm.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &res.proj,
                    self.tp,
                    at(l, "self_attention_res_proj.weight"),
                )?);
            }
            match &w.mixer {
                Mixer::Mla(a) => self.mla(src, &mut tensors, l, a)?,
                Mixer::Kda(k) => self.kda(src, &mut tensors, l, k)?,
            }
            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        self.tp,
                        [at(l, "mlp.gate_proj.weight"), at(l, "mlp.up_proj.weight")],
                    )?);
                    tensors.push(copy(src, down, self.tp, at(l, "mlp.down_proj.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    experts,
                    ..
                } => {
                    tensors.push(copy(
                        src,
                        router,
                        self.tp,
                        at(l, "block_sparse_moe.gate.weight"),
                    )?);
                    tensors.push(self.expert_bank(
                        src,
                        gate_up,
                        (0..*experts).flat_map(|e| {
                            [
                                at(l, &format!("block_sparse_moe.experts.{e}.w1.weight")),
                                at(l, &format!("block_sparse_moe.experts.{e}.w3.weight")),
                            ]
                        }),
                    )?);
                    tensors.push(
                        self.expert_bank(
                            src,
                            down,
                            (0..*experts)
                                .map(|e| at(l, &format!("block_sparse_moe.experts.{e}.w2.weight"))),
                        )?,
                    );
                    if let Some(s) = shared {
                        tensors.push(fused(
                            src,
                            &s.gate_up,
                            self.tp,
                            [
                                at(l, "block_sparse_moe.shared_expert.gate_proj.weight"),
                                at(l, "block_sparse_moe.shared_expert.up_proj.weight"),
                            ],
                        )?);
                        tensors.push(copy(
                            src,
                            &s.down,
                            self.tp,
                            at(l, "block_sparse_moe.shared_expert.down_proj.weight"),
                        )?);
                    }
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
        let mut tensors = Vec::new();
        tensors.push(copy(src, &self.embed, self.tp, GGUF_EMBED)?);
        tensors.push(copy(src, &self.final_norm, self.tp, "output_norm.weight")?);
        tensors.push(copy(src, &self.head, self.tp, "output.weight")?);
        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(
                src,
                &w.mixer_norm,
                self.tp,
                blk(l, "attn_norm.weight"),
            )?);
            tensors.push(copy(src, &w.mlp_norm, self.tp, blk(l, "ffn_norm.weight"))?);
            if let Some(res) = &w.res_blend {
                tensors.push(copy(
                    src,
                    &res.norm,
                    self.tp,
                    blk(l, "attn_res_norm.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &res.proj,
                    self.tp,
                    blk(l, "attn_res_proj.weight"),
                )?);
            }
            match &w.mixer {
                Mixer::Mla(a) => self.gguf_mla(src, &mut tensors, l, a)?,
                Mixer::Kda(k) => self.gguf_kda(src, &mut tensors, l, k)?,
            }
            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        self.tp,
                        [blk(l, "ffn_gate.weight"), blk(l, "ffn_up.weight")],
                    )?);
                    tensors.push(copy(src, down, self.tp, blk(l, "ffn_down.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    ..
                } => {
                    tensors.push(copy(src, router, self.tp, blk(l, "ffn_gate_inp.weight"))?);
                    tensors.push(declare(
                        src,
                        gate_up,
                        self.tp,
                        Expr::concat(
                            1,
                            vec![
                                Expr::src(blk(l, "ffn_gate_exps.weight")),
                                Expr::src(blk(l, "ffn_up_exps.weight")),
                            ],
                        ),
                    )?);
                    tensors.push(copy(src, down, self.tp, blk(l, "ffn_down_exps.weight"))?);
                    if let Some(s) = shared {
                        tensors.push(fused(
                            src,
                            &s.gate_up,
                            self.tp,
                            [
                                blk(l, "ffn_gate_shexp.weight"),
                                blk(l, "ffn_up_shexp.weight"),
                            ],
                        )?);
                        tensors.push(copy(
                            src,
                            &s.down,
                            self.tp,
                            blk(l, "ffn_down_shexp.weight"),
                        )?);
                    }
                }
            }
        }
        Ok(ModelContract {
            alignment: 256,
            tensors,
            groups: Vec::new(),
        })
    }

    fn mla(
        &self,
        src: &ztensor::Source,
        tensors: &mut Vec<TensorContract>,
        l: usize,
        a: &Mla,
    ) -> Result<(), ModelError> {
        tensors.push(copy(
            src,
            &a.q_a_proj,
            self.tp,
            at(l, "self_attn.q_a_proj.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.q_a_norm,
            self.tp,
            at(l, "self_attn.q_a_layernorm.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.q_b_proj,
            self.tp,
            at(l, "self_attn.q_b_proj.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.kv_a_proj,
            self.tp,
            at(l, "self_attn.kv_a_proj_with_mqa.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.kv_a_norm,
            self.tp,
            at(l, "self_attn.kv_a_layernorm.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.kv_b_proj,
            self.tp,
            at(l, "self_attn.kv_b_proj.weight"),
        )?);
        if let Some(gate) = &a.gate {
            tensors.push(copy(src, gate, self.tp, at(l, "self_attn.g_proj.weight"))?);
        }
        tensors.push(copy(
            src,
            &a.o_proj,
            self.tp,
            at(l, "self_attn.o_proj.weight"),
        )?);
        Ok(())
    }

    fn kda(
        &self,
        src: &ztensor::Source,
        tensors: &mut Vec<TensorContract>,
        l: usize,
        k: &Kda,
    ) -> Result<(), ModelError> {
        tensors.push(fused(
            src,
            &k.qkv,
            self.tp,
            [
                at(l, "self_attn.q_proj.weight"),
                at(l, "self_attn.k_proj.weight"),
                at(l, "self_attn.v_proj.weight"),
            ],
        )?);
        tensors.push(fused(
            src,
            &k.conv,
            self.tp,
            [
                at(l, "self_attn.q_conv1d.weight"),
                at(l, "self_attn.k_conv1d.weight"),
                at(l, "self_attn.v_conv1d.weight"),
            ],
        )?);
        tensors.push(copy(
            src,
            &k.f_a,
            self.tp,
            at(l, "self_attn.f_a_proj.weight"),
        )?);
        tensors.push(copy(
            src,
            &k.f_b,
            self.tp,
            at(l, "self_attn.f_b_proj.weight"),
        )?);
        tensors.push(copy(src, &k.b, self.tp, at(l, "self_attn.b_proj.weight"))?);
        tensors.push(copy(src, &k.dt_bias, self.tp, at(l, "self_attn.dt_bias"))?);
        tensors.push(copy(src, &k.a_log, self.tp, at(l, "self_attn.A_log"))?);
        tensors.push(copy(
            src,
            &k.gate,
            self.tp,
            at(l, "self_attn.g_proj.weight"),
        )?);
        tensors.push(copy(
            src,
            &k.o_norm,
            self.tp,
            at(l, "self_attn.o_norm.weight"),
        )?);
        tensors.push(copy(
            src,
            &k.o_proj,
            self.tp,
            at(l, "self_attn.o_proj.weight"),
        )?);
        Ok(())
    }

    fn gguf_mla(
        &self,
        src: &ztensor::Source,
        tensors: &mut Vec<TensorContract>,
        l: usize,
        a: &Mla,
    ) -> Result<(), ModelError> {
        tensors.push(copy(src, &a.q_a_proj, self.tp, blk(l, "attn_q_a.weight"))?);
        tensors.push(copy(
            src,
            &a.q_a_norm,
            self.tp,
            blk(l, "attn_q_a_norm.weight"),
        )?);
        tensors.push(copy(src, &a.q_b_proj, self.tp, blk(l, "attn_q_b.weight"))?);
        tensors.push(copy(
            src,
            &a.kv_a_proj,
            self.tp,
            blk(l, "attn_kv_a_mqa.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.kv_a_norm,
            self.tp,
            blk(l, "attn_kv_a_norm.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.kv_b_proj,
            self.tp,
            blk(l, "attn_kv_b.weight"),
        )?);
        if let Some(gate) = &a.gate {
            tensors.push(copy(src, gate, self.tp, blk(l, "attn_gate.weight"))?);
        }
        tensors.push(copy(src, &a.o_proj, self.tp, blk(l, "attn_output.weight"))?);
        Ok(())
    }

    fn gguf_kda(
        &self,
        src: &ztensor::Source,
        tensors: &mut Vec<TensorContract>,
        l: usize,
        k: &Kda,
    ) -> Result<(), ModelError> {
        tensors.push(copy(src, &k.qkv, self.tp, blk(l, "ssm_in.weight"))?);
        tensors.push(copy(src, &k.conv, self.tp, blk(l, "ssm_conv1d.weight"))?);
        tensors.push(copy(src, &k.f_a, self.tp, blk(l, "ssm_f_a.weight"))?);
        tensors.push(copy(src, &k.f_b, self.tp, blk(l, "ssm_f_b.weight"))?);
        tensors.push(copy(src, &k.b, self.tp, blk(l, "ssm_beta.weight"))?);
        tensors.push(copy(src, &k.dt_bias, self.tp, blk(l, "ssm_dt.bias"))?);
        tensors.push(copy(src, &k.a_log, self.tp, blk(l, "ssm_a"))?);
        tensors.push(copy(src, &k.gate, self.tp, blk(l, "ssm_gate.weight"))?);
        tensors.push(copy(src, &k.o_norm, self.tp, blk(l, "ssm_norm.weight"))?);
        tensors.push(copy(src, &k.o_proj, self.tp, blk(l, "ssm_out.weight"))?);
        Ok(())
    }

    fn expert_bank(
        &self,
        src: &ztensor::Source,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
    ) -> Result<TensorContract, ModelError> {
        let legs = parts.into_iter().map(Expr::src).collect();
        declare(
            src,
            w,
            self.tp,
            Expr::concat(0, legs).transmute(TensorType::raw(lifted(w), DType::BF16)),
        )
    }
}

fn at(l: usize, leaf: &str) -> String {
    format!("language_model.model.layers.{l}.{leaf}")
}

fn blk(l: usize, leaf: &str) -> String {
    format!("blk.{l}.{leaf}")
}

fn lifted(w: &Weight) -> Vec<i64> {
    let mut dims: Vec<i64> = w
        .shape
        .iter()
        .map(|&extent| i64::try_from(extent).expect("an extent no i64 holds"))
        .collect();
    let axis = cut_axis(w);
    let dim = dims.get_mut(axis).unwrap_or_else(|| {
        panic!(
            "`{}` is {:?} and its cut names axis {axis}",
            w.name, w.shape
        )
    });
    *dim = -1;
    dims
}

fn cut_axis(w: &Weight) -> usize {
    match &w.shard {
        Shard::Replicated => panic!("`{}` is replicated and has no cut axis", w.name),
        Shard::Cut { axis, .. } => usize::try_from(*axis).expect("an axis inside a shape"),
    }
}
