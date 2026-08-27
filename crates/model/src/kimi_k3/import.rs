use model_dsl::{Shard, Weight};
use model_loader::contract::{Expr, ModelContract, TensorContract, TensorType};
use model_loader::types::Encoding;

use super::model::{Kda, Mixer, Mla, Mlp, Model};
use crate::contract::{ModelError, copy, declare, fused};

const HF_EMBED: &str = "language_model.model.embed_tokens.weight";

const GGUF_EMBED: &str = "token_embd.weight";

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        if src.get(HF_EMBED).is_some() {
            self.import_from_huggingface(src)
        } else if src.get(GGUF_EMBED).is_some() {
            self.import_from_gguf(src)
        } else {
            Err(ModelError::Illegible {
                name: "kimi_k3".to_string(),
                detail: format!(
                    "neither `{HF_EMBED}` nor `{GGUF_EMBED}` is here, so it names \
                     no format this family reads"
                ),
            })
        }
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let mut tensors = Vec::new();
        tensors.push(copy(src, &self.embed, HF_EMBED)?);
        tensors.push(copy(
            src,
            &self.final_norm,
            "language_model.model.norm.weight",
        )?);
        tensors.push(copy(src, &self.head, "language_model.lm_head.weight")?);
        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(src, &w.mixer_norm, at(l, "input_layernorm.weight"))?);
            tensors.push(copy(
                src,
                &w.mlp_norm,
                at(l, "post_attention_layernorm.weight"),
            )?);
            if let Some(res) = &w.res_blend {
                tensors.push(copy(
                    src,
                    &res.norm,
                    at(l, "self_attention_res_norm.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &res.proj,
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
                        [at(l, "mlp.gate_proj.weight"), at(l, "mlp.up_proj.weight")],
                    )?);
                    tensors.push(copy(src, down, at(l, "mlp.down_proj.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    experts,
                    ..
                } => {
                    tensors.push(copy(src, router, at(l, "block_sparse_moe.gate.weight"))?);
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
                            [
                                at(l, "block_sparse_moe.shared_expert.gate_proj.weight"),
                                at(l, "block_sparse_moe.shared_expert.up_proj.weight"),
                            ],
                        )?);
                        tensors.push(copy(
                            src,
                            &s.down,
                            at(l, "block_sparse_moe.shared_expert.down_proj.weight"),
                        )?);
                    }
                }
            }
        }
        Ok(ModelContract {
            alignment: crate::contract::ALIGNMENT,
            tensors,
            groups: Vec::new(),
        })
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let mut tensors = Vec::new();
        tensors.push(copy(src, &self.embed, GGUF_EMBED)?);
        tensors.push(copy(src, &self.final_norm, "output_norm.weight")?);
        tensors.push(copy(src, &self.head, "output.weight")?);
        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(src, &w.mixer_norm, blk(l, "attn_norm.weight"))?);
            tensors.push(copy(src, &w.mlp_norm, blk(l, "ffn_norm.weight"))?);
            if let Some(res) = &w.res_blend {
                tensors.push(copy(src, &res.norm, blk(l, "attn_res_norm.weight"))?);
                tensors.push(copy(src, &res.proj, blk(l, "attn_res_proj.weight"))?);
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
                        [blk(l, "ffn_gate.weight"), blk(l, "ffn_up.weight")],
                    )?);
                    tensors.push(copy(src, down, blk(l, "ffn_down.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    ..
                } => {
                    tensors.push(copy(src, router, blk(l, "ffn_gate_inp.weight"))?);
                    tensors.push(fused(
                        src,
                        gate_up,
                        [blk(l, "ffn_gate_exps.weight"), blk(l, "ffn_up_exps.weight")],
                    )?);
                    tensors.push(copy(src, down, blk(l, "ffn_down_exps.weight"))?);
                    if let Some(s) = shared {
                        tensors.push(fused(
                            src,
                            &s.gate_up,
                            [
                                blk(l, "ffn_gate_shexp.weight"),
                                blk(l, "ffn_up_shexp.weight"),
                            ],
                        )?);
                        tensors.push(copy(src, &s.down, blk(l, "ffn_down_shexp.weight"))?);
                    }
                }
            }
        }
        Ok(ModelContract {
            alignment: crate::contract::ALIGNMENT,
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
        tensors.push(copy(src, &a.q_a_proj, at(l, "self_attn.q_a_proj.weight"))?);
        tensors.push(copy(
            src,
            &a.q_a_norm,
            at(l, "self_attn.q_a_layernorm.weight"),
        )?);
        tensors.push(copy(src, &a.q_b_proj, at(l, "self_attn.q_b_proj.weight"))?);
        tensors.push(copy(
            src,
            &a.kv_a_proj,
            at(l, "self_attn.kv_a_proj_with_mqa.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.kv_a_norm,
            at(l, "self_attn.kv_a_layernorm.weight"),
        )?);
        tensors.push(copy(
            src,
            &a.kv_b_proj,
            at(l, "self_attn.kv_b_proj.weight"),
        )?);
        if let Some(gate) = &a.gate {
            tensors.push(copy(src, gate, at(l, "self_attn.g_proj.weight"))?);
        }
        tensors.push(copy(src, &a.o_proj, at(l, "self_attn.o_proj.weight"))?);
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
            [
                at(l, "self_attn.q_proj.weight"),
                at(l, "self_attn.k_proj.weight"),
                at(l, "self_attn.v_proj.weight"),
            ],
        )?);
        // SQUEEZED, THEN FUSED, and the squeeze is not optional. HF stores a
        // depthwise convolution bank `[channels, 1, kernel]` — the singleton
        // is `groups`, and PyTorch's Conv1d wants it — while this model
        // declares one rank-2 `[3 * kda_width, kernel]` bank. `fused` alone
        // concatenates three rank-3 legs into a rank-3 expression and the
        // declaration it is checked against has rank 2, so the contract is
        // refused at compile time rather than landing anything wrong. Same
        // fact qwen_3's `squeezed` states for its own KDA conv, with the same
        // per-family copy the redundancy ruling asks for; the GGUF path below
        // needs none, because gguf ships the bank pre-squeezed.
        tensors.push(declare(
            src,
            &k.conv,
            Expr::concat(
                as_axis(cut_axis(&k.conv), &k.conv.name),
                vec![
                    squeezed(src, at(l, "self_attn.q_conv1d.weight"))?,
                    squeezed(src, at(l, "self_attn.k_conv1d.weight"))?,
                    squeezed(src, at(l, "self_attn.v_conv1d.weight"))?,
                ],
            ),
        )?);
        tensors.push(copy(src, &k.f_a, at(l, "self_attn.f_a_proj.weight"))?);
        tensors.push(copy(src, &k.f_b, at(l, "self_attn.f_b_proj.weight"))?);
        tensors.push(copy(src, &k.b, at(l, "self_attn.b_proj.weight"))?);
        tensors.push(copy(src, &k.dt_bias, at(l, "self_attn.dt_bias"))?);
        tensors.push(copy(src, &k.a_log, at(l, "self_attn.A_log"))?);
        tensors.push(copy(src, &k.gate, at(l, "self_attn.g_proj.weight"))?);
        tensors.push(copy(src, &k.o_norm, at(l, "self_attn.o_norm.weight"))?);
        tensors.push(copy(src, &k.o_proj, at(l, "self_attn.o_proj.weight"))?);
        Ok(())
    }

    fn gguf_mla(
        &self,
        src: &ztensor::Source,
        tensors: &mut Vec<TensorContract>,
        l: usize,
        a: &Mla,
    ) -> Result<(), ModelError> {
        tensors.push(copy(src, &a.q_a_proj, blk(l, "attn_q_a.weight"))?);
        tensors.push(copy(src, &a.q_a_norm, blk(l, "attn_q_a_norm.weight"))?);
        tensors.push(copy(src, &a.q_b_proj, blk(l, "attn_q_b.weight"))?);
        tensors.push(copy(src, &a.kv_a_proj, blk(l, "attn_kv_a_mqa.weight"))?);
        tensors.push(copy(src, &a.kv_a_norm, blk(l, "attn_kv_a_norm.weight"))?);
        tensors.push(copy(src, &a.kv_b_proj, blk(l, "attn_kv_b.weight"))?);
        if let Some(gate) = &a.gate {
            tensors.push(copy(src, gate, blk(l, "attn_gate.weight"))?);
        }
        tensors.push(copy(src, &a.o_proj, blk(l, "attn_output.weight"))?);
        Ok(())
    }

    fn gguf_kda(
        &self,
        src: &ztensor::Source,
        tensors: &mut Vec<TensorContract>,
        l: usize,
        k: &Kda,
    ) -> Result<(), ModelError> {
        tensors.push(copy(src, &k.qkv, blk(l, "ssm_in.weight"))?);
        tensors.push(copy(src, &k.conv, blk(l, "ssm_conv1d.weight"))?);
        tensors.push(copy(src, &k.f_a, blk(l, "ssm_f_a.weight"))?);
        tensors.push(copy(src, &k.f_b, blk(l, "ssm_f_b.weight"))?);
        tensors.push(copy(src, &k.b, blk(l, "ssm_beta.weight"))?);
        tensors.push(copy(src, &k.dt_bias, blk(l, "ssm_dt.bias"))?);
        tensors.push(copy(src, &k.a_log, blk(l, "ssm_a"))?);
        tensors.push(copy(src, &k.gate, blk(l, "ssm_gate.weight"))?);
        tensors.push(copy(src, &k.o_norm, blk(l, "ssm_norm.weight"))?);
        tensors.push(copy(src, &k.o_proj, blk(l, "ssm_out.weight"))?);
        Ok(())
    }

    fn expert_bank(
        &self,
        src: &ztensor::Source,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
    ) -> Result<TensorContract, ModelError> {
        let names: Vec<String> = parts.into_iter().collect();
        let first = names
            .first()
            .expect("an expert bank stacks at least one leg");
        let read = match crate::contract::stored_encoding(src, first)? {
            Encoding::Raw(dtype) => dtype,
            Encoding::Quant(spec) => spec.logical_dtype,
        };
        let legs = names.into_iter().map(Expr::src).collect();
        let stack = TensorType::raw(lifted(w, cut_axis(w)), read);
        declare(src, w, Expr::concat(0, legs).transmute(stack))
    }
}

/// One depthwise convolution bank, with the singleton `groups` axis dropped.
///
/// A `Transmute` and not a reshape node: the bytes of `[channels, 1, kernel]`
/// and `[channels, kernel]` are the same bytes in the same order, so this
/// renames the type and moves nothing. The stored encoding is read off the
/// tensor rather than assumed, because a transmute may not change what an
/// element means.
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

fn as_axis(axis: usize, name: &str) -> u8 {
    u8::try_from(axis)
        .unwrap_or_else(|_| panic!("`{name}` is packed on axis {axis}, which is no axis"))
}

fn at(l: usize, leaf: &str) -> String {
    format!("language_model.model.layers.{l}.{leaf}")
}

fn blk(l: usize, leaf: &str) -> String {
    format!("blk.{l}.{leaf}")
}

fn lifted(w: &Weight, axis: usize) -> Vec<i64> {
    let mut dims: Vec<i64> = w
        .shape
        .iter()
        .map(|&extent| i64::try_from(extent).expect("an extent no i64 holds"))
        .collect();
    let dim = dims.get_mut(axis).unwrap_or_else(|| {
        panic!(
            "`{}` is {:?} and the stack's wildcard names axis {axis}",
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
