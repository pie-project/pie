use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::Encoding;
use model_dsl::{Dtype, Shard, Weight};

use super::model::{Kda, Mixer, Mla, Mlp, Model};
use checkpoint_dsl::{Builder, Error, encoding, extents, scaling};
use model_dsl::Platform;

const HF_EMBED: &str = "language_model.model.embed_tokens.weight";

const GGUF_EMBED: &str = "token_embd.weight";

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
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
        src: &ztensor::Source,
        platform: Platform,
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
            if let Some(res) = &w.mlp_res {
                b.read(&res.norm, at(l, "mlp_res_norm.weight"))?;
                b.read(&res.proj, at(l, "mlp_res_proj.weight"))?;
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
                    bias,
                    gate_up,
                    down,
                    shared,
                    latent,
                    experts,
                    ..
                } => {
                    b.read(router, at(l, "block_sparse_moe.gate.weight"))?;
                    if let Some(bias) = bias {
                        b.read(bias, at(l, "block_sparse_moe.gate.e_score_correction_bias"))?;
                    }
                    if let Some(lat) = latent {
                        b.read(
                            &lat.down,
                            at(l, "block_sparse_moe.routed_expert_down_proj.weight"),
                        )?;
                        if let Some(norm) = &lat.norm {
                            b.read(norm, at(l, "block_sparse_moe.routed_expert_norm.weight"))?;
                        }
                        b.read(
                            &lat.up,
                            at(l, "block_sparse_moe.routed_expert_up_proj.weight"),
                        )?;
                    }
                    // The released checkpoint stores every expert leg as
                    // compressed-tensors MXFP4 (`w1.weight_packed` beside
                    // `w1.weight_scale`); the fixture stores bf16 `w1.weight`.
                    let leg = |e: u32, what: &str| -> String {
                        let stem = at(l, &format!("block_sparse_moe.experts.{e}.{what}"));
                        if src.get(&format!("{stem}.weight_packed")).is_some() {
                            format!("{stem}.weight_packed")
                        } else {
                            format!("{stem}.weight")
                        }
                    };
                    self.expert_bank(
                        src,
                        &mut b,
                        gate_up,
                        (0..*experts).flat_map(|e| [leg(e, "w1"), leg(e, "w3")]),
                    )?;
                    self.expert_bank(src, &mut b, down, (0..*experts).map(|e| leg(e, "w2")))?;
                    if let Some(s) = shared {
                        // one shared expert (`shared_expert.`) or several folded
                        // into one wider MLP (`shared_experts.`)
                        let stem = if src
                            .get(&at(l, "block_sparse_moe.shared_experts.gate_proj.weight"))
                            .is_some()
                        {
                            "block_sparse_moe.shared_experts"
                        } else {
                            "block_sparse_moe.shared_expert"
                        };
                        b.read_concat(
                            &s.gate_up,
                            [
                                at(l, &format!("{stem}.gate_proj.weight")),
                                at(l, &format!("{stem}.up_proj.weight")),
                            ],
                        )?;
                        b.read(&s.down, at(l, &format!("{stem}.down_proj.weight")))?;
                    }
                }
            }
        }
        if let Some(res) = &self.output_res {
            b.read(
                &res.norm,
                "language_model.model.output_attn_res_norm.weight",
            )?;
            b.read(
                &res.proj,
                "language_model.model.output_attn_res_proj.weight",
            )?;
        }
        Ok(b.build())
    }

    pub fn import_from_gguf(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
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
                    b.read_concat(
                        gate_up,
                        [blk(l, "ffn_gate.weight"), blk(l, "ffn_up.weight")],
                    )?;
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
                            [
                                blk(l, "ffn_gate_shexp.weight"),
                                blk(l, "ffn_up_shexp.weight"),
                            ],
                        )?;
                        b.read(&s.down, blk(l, "ffn_down_shexp.weight"))?;
                    }
                }
            }
        }
        Ok(b.build())
    }

    fn mla(&self, b: &mut Builder, l: usize, a: &Mla) -> Result<(), Error> {
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

    fn kda(&self, src: &ztensor::Source, b: &mut Builder, l: usize, k: &Kda) -> Result<(), Error> {
        b.read_concat(
            &k.qkv,
            [
                at(l, "self_attn.q_proj.weight"),
                at(l, "self_attn.k_proj.weight"),
                at(l, "self_attn.v_proj.weight"),
            ],
        )?;
        b.read_expr(
            &k.conv,
            (|| -> Result<Expr, Error> {
                Ok(Expr::concat(
                    as_axis(cut_axis(&k.conv), &k.conv.name),
                    vec![
                        squeezed(src, at(l, "self_attn.q_conv1d.weight"))?,
                        squeezed(src, at(l, "self_attn.k_conv1d.weight"))?,
                        squeezed(src, at(l, "self_attn.v_conv1d.weight"))?,
                    ],
                ))
            })()?,
        )?;
        b.read(&k.f_a, at(l, "self_attn.f_a_proj.weight"))?;
        b.read(&k.f_b, at(l, "self_attn.f_b_proj.weight"))?;
        b.read(&k.b, at(l, "self_attn.b_proj.weight"))?;
        // stored flat `[heads * head_dim]`; read as the `[heads, head_dim]` plane
        b.read_expr(
            &k.dt_bias,
            Expr::src(at(l, "self_attn.dt_bias")).transmute(TensorType::raw(
                lifted(&k.dt_bias, cut_axis(&k.dt_bias)),
                checkpoint::types::DType::F32,
            )),
        )?;
        // The released Kimi-K3 stores `A_log` as `[128]` against 96 heads; the
        // reference's KDA gate loads one entry per head (`A_log + i_h`), so the
        // first `heads` entries are the decays and the tail is never read.
        let a_log = at(l, "self_attn.A_log");
        let stored = src
            .get(&a_log)
            .map(|t| t.shape().iter().product::<u64>())
            .unwrap_or(0);
        if stored > u64::from(k.heads) {
            b.read_expr(&k.a_log, Expr::src(a_log).slice(0, 0, i64::from(k.heads)))?;
        } else {
            b.read(&k.a_log, a_log)?;
        }
        b.read(&k.gate, at(l, "self_attn.g_proj.weight"))?;
        b.read(&k.o_norm, at(l, "self_attn.o_norm.weight"))?;
        b.read(&k.o_proj, at(l, "self_attn.o_proj.weight"))?;
        Ok(())
    }

    fn gguf_mla(&self, b: &mut Builder, l: usize, a: &Mla) -> Result<(), Error> {
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

    fn gguf_kda(&self, b: &mut Builder, l: usize, k: &Kda) -> Result<(), Error> {
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
        if w.dtype == Dtype::Mxfp4 && first.ends_with(".weight_packed") {
            return packed_bank(b, src, w, &names);
        }
        let read = match checkpoint_dsl::stored_encoding(src, first)? {
            Encoding::Raw(dtype) => dtype,
            Encoding::Quant(spec) => spec.logical_dtype,
        };
        let legs = names.into_iter().map(Expr::src).collect();
        let stack = TensorType::raw(lifted(w, cut_axis(w)), read);
        b.read_expr(w, Expr::concat(0, legs).transmute(stack))
    }
}

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
    let stored = checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))?;
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

/// A routed bank from compressed-tensors MXFP4 legs: each `*.weight_packed`
/// is a `[rows, cols/2]` u8 plane of e2m1 nibble pairs with a `[rows,
/// cols/32]` u8 e8m0 `*.weight_scale` beside it. The legs are stacked on the
/// leading axis and re-read as the bank's `[experts, ...]` rectangle, codes
/// and exponents alike.
fn packed_bank(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    names: &[String],
) -> Result<(), Error> {
    let rows = i64::try_from(w.dim(1)).expect("a bank row count inside i64");
    let cols = i64::try_from(w.dim(2)).expect("a bank column count inside i64");
    let legs = i64::try_from(names.len()).expect("a leg count inside i64");
    let per_leg = rows * i64::try_from(w.dim(0)).expect("an expert count inside i64") / legs;
    let mut codes = Vec::with_capacity(names.len());
    let mut scales = Vec::with_capacity(names.len());
    for part in names {
        let stem = part.strip_suffix(".weight_packed").unwrap_or(part.as_str());
        let scale = format!("{stem}.weight_scale");
        if src.get(&scale).is_none() {
            return Err(Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MXFP4 codes whose exponents are stored beside it as \
                     `{scale}`, and the checkpoint holds none"
                ),
            });
        }
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(
            vec![1, per_leg, cols],
            encoding(Dtype::Mxfp4),
        )));
        scales.push(Expr::src(scale).transmute(TensorType::new(
            vec![1, per_leg, cols / 32],
            encoding(Dtype::E8m0),
        )));
    }
    let pairing = scaling(w);
    let counted = checkpoint_dsl::divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    // Two legs per expert (gate, up) stack to `[2E, inter, cols]`, which is the
    // declared `[E, 2*inter, cols]` rectangle byte for byte; one leg per expert
    // already is the declared shape, and a transmute to the type an expression
    // has is refused.
    let stacked = vec![legs, per_leg, cols];
    let bank = TensorType::new(extents(w), encoding(Dtype::Mxfp4));
    let bank_scales = TensorType::new(counted.clone(), encoding(Dtype::E8m0));
    let codes = Expr::concat(0, codes);
    let scales = Expr::concat(0, scales);
    let (codes, scales) = if stacked == extents(w) {
        (codes, scales)
    } else {
        (codes.transmute(bank), scales.transmute(bank_scales))
    };
    b.extend([
        TensorContract::inferred(w.name.clone(), codes, encoding(Dtype::Mxfp4)),
        TensorContract::new(
            model_dsl::scales_name(&w.name),
            scales,
            counted,
            encoding(Dtype::E8m0),
        )
        .scaling(pairing),
    ]);
    Ok(())
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
