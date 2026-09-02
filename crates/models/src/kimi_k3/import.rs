use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint::types::Encoding;
use model_dsl::{Shard, Weight};

use super::model::{Kda, Mixer, Mla, Mlp, Model};
use model_dsl::Platform;
use checkpoint_dsl::{Builder, Error};

const HF_EMBED: &str = "language_model.model.embed_tokens.weight";

const GGUF_EMBED: &str = "token_embd.weight";

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        // Try the native (already-imported) layout first.
        let huggingface = match self.import_from_huggingface(src, platform) {
            Ok(contract) => return Ok(contract),
            Err(why) => why,
        };
        let gguf = match self.import_from_gguf(src, platform) {
            Ok(contract) => return Ok(contract),
            Err(why) => why,
        };
        Err(Error::Illegible {
            name: "kimi_k3".to_string(),
            detail: format!(
                "no reading of this file lands every plane this family \
                 declares — as huggingface, {huggingface}; as gguf, {gguf}"
            ),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source, platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, HF_EMBED)?;
        b.read(&self.final_norm, "language_model.model.norm.weight")?;
        b.read(&self.head, "language_model.lm_head.weight")?;
        for (l, w) in self.layers.iter().enumerate() {
            b.read(&w.mixer_norm, at(l, "input_layernorm.weight"))?;
            b.read(&w.mlp_norm, at(l, "post_attention_layernorm.weight"))?;
            if let Some(res) = &w.res_blend {
                b.read(&res.norm, at(l, "self_attention_res_norm.weight"))?;
                b.read(&res.proj, at(l, "self_attention_res_proj.weight"))?;
            }
            match &w.mixer {
                Mixer::Mla(a) => self.mla(&mut b, l, a)?,
                Mixer::Kda(k) => self.kda(src, &mut b, l, k)?,
            }
            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(
                        gate_up,
                        [at(l, "mlp.gate_proj.weight"), at(l, "mlp.up_proj.weight")],
                    )?;
                    b.read(down, at(l, "mlp.down_proj.weight"))?;
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    experts,
                    ..
                } => {
                    b.read(router, at(l, "block_sparse_moe.gate.weight"))?;
                    self.expert_bank(
                        src,
                        &mut b,
                        gate_up,
                        (0..*experts).flat_map(|e| {
                            [
                                at(l, &format!("block_sparse_moe.experts.{e}.w1.weight")),
                                at(l, &format!("block_sparse_moe.experts.{e}.w3.weight")),
                            ]
                        }),
                    )?;
                    self.expert_bank(
                        src,
                        &mut b,
                        down,
                        (0..*experts)
                            .map(|e| at(l, &format!("block_sparse_moe.experts.{e}.w2.weight"))),
                    )?;
                    if let Some(s) = shared {
                        b.read_concat(
                            &s.gate_up,
                            [
                                at(l, "block_sparse_moe.shared_expert.gate_proj.weight"),
                                at(l, "block_sparse_moe.shared_expert.up_proj.weight"),
                            ],
                        )?;
                        b.read(&s.down, at(l, "block_sparse_moe.shared_expert.down_proj.weight"))?;
                    }
                }
            }
        }
        Ok(b.build())
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source, platform: Platform) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, GGUF_EMBED)?;
        b.read(&self.final_norm, "output_norm.weight")?;
        b.read(&self.head, "output.weight")?;
        for (l, w) in self.layers.iter().enumerate() {
            b.read(&w.mixer_norm, blk(l, "attn_norm.weight"))?;
            b.read(&w.mlp_norm, blk(l, "ffn_norm.weight"))?;
            if let Some(res) = &w.res_blend {
                b.read(&res.norm, blk(l, "attn_res_norm.weight"))?;
                b.read(&res.proj, blk(l, "attn_res_proj.weight"))?;
            }
            match &w.mixer {
                Mixer::Mla(a) => self.gguf_mla(&mut b, l, a)?,
                Mixer::Kda(k) => self.gguf_kda(&mut b, l, k)?,
            }
            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(gate_up, [blk(l, "ffn_gate.weight"), blk(l, "ffn_up.weight")])?;
                    b.read(down, blk(l, "ffn_down.weight"))?;
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    ..
                } => {
                    b.read(router, blk(l, "ffn_gate_inp.weight"))?;
                    b.read_concat(
                        gate_up,
                        [blk(l, "ffn_gate_exps.weight"), blk(l, "ffn_up_exps.weight")],
                    )?;
                    b.read(down, blk(l, "ffn_down_exps.weight"))?;
                    if let Some(s) = shared {
                        b.read_concat(
                            &s.gate_up,
                            [blk(l, "ffn_gate_shexp.weight"), blk(l, "ffn_up_shexp.weight")],
                        )?;
                        b.read(&s.down, blk(l, "ffn_down_shexp.weight"))?;
                    }
                }
            }
        }
        Ok(b.build())
    }

    fn mla(
        &self,
        b: &mut Builder,
        l: usize,
        a: &Mla,
    ) -> Result<(), Error> {
        b.read(&a.q_a_proj, at(l, "self_attn.q_a_proj.weight"))?;
        b.read(&a.q_a_norm, at(l, "self_attn.q_a_layernorm.weight"))?;
        b.read(&a.q_b_proj, at(l, "self_attn.q_b_proj.weight"))?;
        b.read(&a.kv_a_proj, at(l, "self_attn.kv_a_proj_with_mqa.weight"))?;
        b.read(&a.kv_a_norm, at(l, "self_attn.kv_a_layernorm.weight"))?;
        b.read(&a.kv_b_proj, at(l, "self_attn.kv_b_proj.weight"))?;
        if let Some(gate) = &a.gate {
            b.read(gate, at(l, "self_attn.g_proj.weight"))?;
        }
        b.read(&a.o_proj, at(l, "self_attn.o_proj.weight"))?;
        Ok(())
    }

    fn kda(
        &self,
        src: &ztensor::Source,
        b: &mut Builder,
        l: usize,
        k: &Kda,
    ) -> Result<(), Error> {
        b.read_concat(
            &k.qkv,
            [
                at(l, "self_attn.q_proj.weight"),
                at(l, "self_attn.k_proj.weight"),
                at(l, "self_attn.v_proj.weight"),
            ],
        )?;
        // HF stores the conv bank as [channels, 1, kernel] (the 1 is `groups`); the declared
        // type is rank-2 [3*kda_width, kernel], so each leg is squeezed before concatenation.
        b.read_expr(&k.conv, (|| -> Result<Expr, Error> {
            Ok(Expr::concat(
                as_axis(cut_axis(&k.conv), &k.conv.name),
                vec![
                    squeezed(src, at(l, "self_attn.q_conv1d.weight"))?,
                    squeezed(src, at(l, "self_attn.k_conv1d.weight"))?,
                    squeezed(src, at(l, "self_attn.v_conv1d.weight"))?,
                ],
            ))
        })()?)?;
        b.read(&k.f_a, at(l, "self_attn.f_a_proj.weight"))?;
        b.read(&k.f_b, at(l, "self_attn.f_b_proj.weight"))?;
        b.read(&k.b, at(l, "self_attn.b_proj.weight"))?;
        b.read(&k.dt_bias, at(l, "self_attn.dt_bias"))?;
        b.read(&k.a_log, at(l, "self_attn.A_log"))?;
        b.read(&k.gate, at(l, "self_attn.g_proj.weight"))?;
        b.read(&k.o_norm, at(l, "self_attn.o_norm.weight"))?;
        b.read(&k.o_proj, at(l, "self_attn.o_proj.weight"))?;
        Ok(())
    }

    fn gguf_mla(
        &self,
        b: &mut Builder,
        l: usize,
        a: &Mla,
    ) -> Result<(), Error> {
        b.read(&a.q_a_proj, blk(l, "attn_q_a.weight"))?;
        b.read(&a.q_a_norm, blk(l, "attn_q_a_norm.weight"))?;
        b.read(&a.q_b_proj, blk(l, "attn_q_b.weight"))?;
        b.read(&a.kv_a_proj, blk(l, "attn_kv_a_mqa.weight"))?;
        b.read(&a.kv_a_norm, blk(l, "attn_kv_a_norm.weight"))?;
        b.read(&a.kv_b_proj, blk(l, "attn_kv_b.weight"))?;
        if let Some(gate) = &a.gate {
            b.read(gate, blk(l, "attn_gate.weight"))?;
        }
        b.read(&a.o_proj, blk(l, "attn_output.weight"))?;
        Ok(())
    }

    fn gguf_kda(
        &self,
        b: &mut Builder,
        l: usize,
        k: &Kda,
    ) -> Result<(), Error> {
        b.read(&k.qkv, blk(l, "ssm_in.weight"))?;
        b.read(&k.conv, blk(l, "ssm_conv1d.weight"))?;
        b.read(&k.f_a, blk(l, "ssm_f_a.weight"))?;
        b.read(&k.f_b, blk(l, "ssm_f_b.weight"))?;
        b.read(&k.b, blk(l, "ssm_beta.weight"))?;
        b.read(&k.dt_bias, blk(l, "ssm_dt.bias"))?;
        b.read(&k.a_log, blk(l, "ssm_a"))?;
        b.read(&k.gate, blk(l, "ssm_gate.weight"))?;
        b.read(&k.o_norm, blk(l, "ssm_norm.weight"))?;
        b.read(&k.o_proj, blk(l, "ssm_out.weight"))?;
        Ok(())
    }

    /// Stacks the source's per-expert w1/w3/w2 legs (stored bf16) into the Mxfp4 bank this
    /// model declares; the bf16→Mxfp4 conversion runs here, at import time.
    fn expert_bank(
        &self,
        src: &ztensor::Source,
        b: &mut Builder,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
    ) -> Result<(), Error> {
        let names: Vec<String> = parts.into_iter().collect();
        let first = names
            .first()
            .expect("an expert bank stacks at least one leg");
        let read = match checkpoint_dsl::stored_encoding(src, first)? {
            Encoding::Raw(dtype) => dtype,
            Encoding::Quant(spec) => spec.logical_dtype,
        };
        let legs = names.into_iter().map(Expr::src).collect();
        let stack = TensorType::raw(lifted(w, cut_axis(w)), read);
        b.read_expr(w, Expr::concat(0, legs).transmute(stack))
    }
}

/// Drops the singleton `groups` axis: [channels, 1, kernel] -> [channels, kernel], same bytes.
fn squeezed(src: &ztensor::Source, from: String) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
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
    let stored = checkpoint::file::encoding_of(&tensor, &part).map_err(|why| illegible(&why))?;
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
