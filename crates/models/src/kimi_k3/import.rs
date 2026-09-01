use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint::types::Encoding;
use model_dsl::{Shard, Weight};

use super::model::{Kda, Mixer, Mla, Mlp, Model};
use checkpoint_dsl::{Builder, Error};

const HF_EMBED: &str = "language_model.model.embed_tokens.weight";

const GGUF_EMBED: &str = "token_embd.weight";

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        // **THE NATIVE DOOR, ASKED BEFORE THE WITNESS SNIFF** (§M-4a). A file
        // holding every plane this contract declares, under this contract's
        // names, is an artifact `pie model import` wrote out of this very
        // text, and [`Model::load`] is its reader: `read_own` throughout, no
        // transform at all. `load` failing is what says the file is foreign,
        // and it fails on the first plane it cannot find. The argument in full
        // is at `qwen_3::Model::import`.
        if let Ok(native) = self.load(src) {
            return Ok(native);
        }
        // **AND THE ARM IS CHOSEN BY BUILDING IT, NOT BY SNIFFING A NAME.**
        // The witness this used to look for — the embedding, spelled the way
        // each layout spells it — is one of the planes a promotion MOVES, so
        // an artifact this build wrote could satisfy neither door. The
        // argument in full, and the file it was measured on, is at
        // `qwen_3::Model::import`.
        let huggingface = match self.import_from_huggingface(src) {
            Ok(contract) => return Ok(contract),
            Err(why) => why,
        };
        let gguf = match self.import_from_gguf(src) {
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
        src: &ztensor::Source,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
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

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
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
        // SQUEEZED, THEN FUSED, and the squeeze is not optional. HF stores a
        // depthwise convolution bank `[channels, 1, kernel]` — the singleton
        // is `groups`, and PyTorch's Conv1d wants it — while this model
        // declares one rank-2 `[3 * kda_width, kernel]` bank. `read_concat`
        // alone concatenates three rank-3 legs into a rank-3 expression and the
        // declaration it is checked against has rank 2, so the contract is
        // refused at compile time rather than landing anything wrong. Same
        // fact qwen_3's `squeezed` states for its own KDA conv, with the same
        // per-family copy the redundancy ruling asks for; the GGUF path below
        // needs none, because gguf ships the bank pre-squeezed.
        b.read_derived(&k.conv, || {
            Ok(Expr::concat(
                as_axis(cut_axis(&k.conv), &k.conv.name),
                vec![
                    squeezed(src, at(l, "self_attn.q_conv1d.weight"))?,
                    squeezed(src, at(l, "self_attn.k_conv1d.weight"))?,
                    squeezed(src, at(l, "self_attn.v_conv1d.weight"))?,
                ],
            ))
        })?;
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

    /// One expert bank, in the SOURCE spelling: `n` separate `w1`/`w3`/`w2`
    /// legs, stored bf16, which this family declares `Mxfp4`. That gap is an
    /// encode, and an encode is a conversion — `pie model import` is where it
    /// runs, and no device target may carry one
    /// ([`CUDA_TILE_MAP_MASK`](checkpoint::plan::CUDA_TILE_MAP_MASK)). So
    /// this states the conversion for the importer to run and is refused
    /// with the tensor named on any serving load that reaches it.
    ///
    /// **THE ARM THAT READ THE ALREADY-STACKED BANK MOVED** (§M-3, §M-4a).
    /// It opened `if src.get(&w.name).is_some() { b.read_own(w) }`: `w.name`
    /// is what an import writes this bank under once it has done the stacking
    /// and the encoding — one plane and its `.scales` companion, under the
    /// model's own spelling — so a file holding that name holds the served
    /// form already. §M-4a promotes every transform a contract states and not
    /// only this one, so the question is now asked of every weight by every
    /// verb: [`Builder::holds_the_landed_plane`]. The reading is unchanged and
    /// the banding a sharded rank needs still comes from `read_own`.
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

/// One depthwise convolution bank, with the singleton `groups` axis dropped.
///
/// A `Transmute` and not a reshape node: the bytes of `[channels, 1, kernel]`
/// and `[channels, kernel]` are the same bytes in the same order, so this
/// renames the type and moves nothing. The stored encoding is read off the
/// tensor rather than assumed, because a transmute may not change what an
/// element means.
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
