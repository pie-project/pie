use checkpoint::contract::{Expr, ModelContract, TensorType, UnaryOp};

use super::model::{Head, Mixer, Mlp, Model};
use checkpoint_dsl::{Builder, Error, extents};
use model_dsl::Platform;

#[derive(Clone, Copy)]
enum Layout {
    Transformers,
    Mlx,
}

impl Layout {
    fn spelling(self) -> &'static str {
        match self {
            Self::Transformers => "transformers",
            Self::Mlx => "mlx_lm",
        }
    }

    fn embed(self) -> &'static str {
        match self {
            Self::Transformers => "model.language_model.embed_tokens.weight",
            Self::Mlx => "language_model.model.embed_tokens.weight",
        }
    }

    fn norm(self) -> &'static str {
        match self {
            Self::Transformers => "model.language_model.norm.weight",
            Self::Mlx => "language_model.model.norm.weight",
        }
    }

    fn head(self) -> &'static str {
        match self {
            Self::Transformers => "lm_head.weight",
            Self::Mlx => "language_model.lm_head.weight",
        }
    }

    fn layer(self, l: usize, leaf: &str) -> String {
        match self {
            Self::Transformers => format!("model.language_model.layers.{l}.{leaf}"),
            Self::Mlx => format!("language_model.model.layers.{l}.{leaf}"),
        }
    }

    fn tower(self, leaf: &str) -> String {
        match self {
            Self::Transformers => format!("model.visual.{leaf}"),
            Self::Mlx => format!("vision_tower.{leaf}"),
        }
    }

    fn folds_the_norm_one(self) -> bool {
        match self {
            Self::Transformers => false,
            Self::Mlx => true,
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut refusals: Vec<String> = Vec::new();
        let mut attempt = |what: &str, built: Result<ModelContract, Error>| match built {
            Ok(contract) => Some(contract),
            Err(why) => {
                refusals.push(format!("as {what}, {why}"));
                None
            }
        };
        for layout in [Layout::Transformers, Layout::Mlx] {
            if let Some(contract) = attempt(
                layout.spelling(),
                self.import_from_safetensors(src, platform, layout),
            ) {
                return Ok(contract);
            }
        }
        if let Some(contract) = attempt("gguf", self.import_from_gguf(src, platform)) {
            return Ok(contract);
        }
        Err(Error::Illegible {
            name: "qwen_3".to_string(),
            detail: format!(
                "no reading of this file lands every plane this family \
                 declares — {}",
                refusals.join("; "),
            ),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from_safetensors(src, platform, Layout::Transformers)
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let norm = |from: String| -> Expr {
            let read = Expr::src(from);
            if layout.folds_the_norm_one() {
                read.bias(-1.0)
            } else {
                read
            }
        };

        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, layout.embed())?;
        b.read_expr(&self.final_norm, norm(layout.norm().to_string()))?;

        if let Head::Bank(head) = &self.head {
            b.read(head, layout.head())?;
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| layout.layer(l, s);

            b.read_expr(&w.mixer_norm, norm(n("input_layernorm.weight")))?;
            b.read_expr(&w.mlp_norm, norm(n("post_attention_layernorm.weight")))?;

            match &w.mixer {
                Mixer::Attn(a) => {
                    b.read(&a.qg_proj, n("self_attn.q_proj.weight"))?;
                    b.read(&a.k_proj, n("self_attn.k_proj.weight"))?;
                    b.read(&a.v_proj, n("self_attn.v_proj.weight"))?;
                    b.read(&a.o_proj, n("self_attn.o_proj.weight"))?;
                    b.read_expr(&a.q_norm, norm(n("self_attn.q_norm.weight")))?;
                    b.read_expr(&a.k_norm, norm(n("self_attn.k_norm.weight")))?;
                }
                Mixer::Gdn(g) => {
                    b.read_concat(
                        &g.in_qkvz,
                        [
                            n("linear_attn.in_proj_qkv.weight"),
                            n("linear_attn.in_proj_z.weight"),
                        ],
                    )?;

                    b.read_concat(
                        &g.in_ba,
                        [
                            n("linear_attn.in_proj_b.weight"),
                            n("linear_attn.in_proj_a.weight"),
                        ],
                    )?;

                    b.read_expr(&g.conv, squeezed(src, n("linear_attn.conv1d.weight"))?)?;

                    b.read(&g.dt_bias, n("linear_attn.dt_bias"))?;
                    b.read(&g.a_log, n("linear_attn.A_log"))?;
                    b.read(&g.norm, n("linear_attn.norm.weight"))?;
                    b.read(&g.out_proj, n("linear_attn.out_proj.weight"))?;
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?;
                    b.read(down, n("mlp.down_proj.weight"))?;
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
                    b.read(router, n("mlp.gate.weight"))?;

                    match layout {
                        Layout::Transformers => {
                            b.read(gate_up, n("mlp.experts.gate_up_proj"))?;
                            b.read(down, n("mlp.experts.down_proj"))?;
                        }
                        Layout::Mlx => {
                            b.read_concat(
                                gate_up,
                                [
                                    n("mlp.switch_mlp.gate_proj.weight"),
                                    n("mlp.switch_mlp.up_proj.weight"),
                                ],
                            )?;
                            b.read(down, n("mlp.switch_mlp.down_proj.weight"))?;
                        }
                    }
                    b.read_concat(
                        shared_gate_up,
                        [
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    )?;
                    b.read(shared_down, n("mlp.shared_expert.down_proj.weight"))?;
                    b.read(shared_gate, n("mlp.shared_expert_gate.weight"))?;
                }
            }
        }

        if let Some(t) = &self.tower {
            let v = |s: &str| layout.tower(s);
            const CHANNELS: i64 = 3;
            let want = extents(&t.patch_embed);
            b.read_expr(
                &t.patch_embed,
                (|| -> Result<Expr, Error> {
                    let flat = flattened(src, v("patch_embed.proj.weight"), want.clone())?;
                    Ok(match layout {
                        Layout::Transformers => flat,
                        Layout::Mlx => {
                            let per = want[1] / CHANNELS;
                            let indices = (0..CHANNELS)
                                .flat_map(|c| (0..per).map(move |j| j * CHANNELS + c))
                                .collect();
                            flat.gather(1, indices)
                        }
                    })
                })()?,
            )?;
            b.read(&t.patch_embed_bias, v("patch_embed.proj.bias"))?;
            b.read(&t.pos_embed, v("pos_embed.weight"))?;
            for (l, blk) in t.blocks.iter().enumerate() {
                let n = |s: &str| v(&format!("blocks.{l}.{s}"));
                for (weight, from) in [
                    (&blk.norm1, n("norm1.weight")),
                    (&blk.norm1_bias, n("norm1.bias")),
                    (&blk.qkv, n("attn.qkv.weight")),
                    (&blk.qkv_bias, n("attn.qkv.bias")),
                    (&blk.proj, n("attn.proj.weight")),
                    (&blk.proj_bias, n("attn.proj.bias")),
                    (&blk.norm2, n("norm2.weight")),
                    (&blk.norm2_bias, n("norm2.bias")),
                    (&blk.fc1, n("mlp.linear_fc1.weight")),
                    (&blk.fc1_bias, n("mlp.linear_fc1.bias")),
                    (&blk.fc2, n("mlp.linear_fc2.weight")),
                    (&blk.fc2_bias, n("mlp.linear_fc2.bias")),
                ] {
                    b.read(weight, from)?;
                }
            }
            let m = &t.merger;
            for (weight, from) in [
                (&m.norm, v("merger.norm.weight")),
                (&m.norm_bias, v("merger.norm.bias")),
                (&m.fc1, v("merger.linear_fc1.weight")),
                (&m.fc1_bias, v("merger.linear_fc1.bias")),
                (&m.fc2, v("merger.linear_fc2.weight")),
                (&m.fc2_bias, v("merger.linear_fc2.bias")),
            ] {
                b.read(weight, from)?;
            }
        }

        if let Some(mtp) = &self.mtp {
            let p: String =
                if src.get("aux.fc.weight").is_some() || src.get("aux.fc_embed.weight").is_some() {
                    "aux".to_string()
                } else {
                    mtp.recipe.prefix().to_string()
                };
            let n = |s: &str| format!("{p}.layers.0.{s}");
            if let Some(pre) = &mtp.pre_fc {
                b.read(&pre.embedding, format!("{p}.pre_fc_norm_embedding.weight"))?;
                b.read(&pre.hidden, format!("{p}.pre_fc_norm_hidden.weight"))?;
            }

            if src.get(&format!("{p}.fc_embed.weight")).is_some() {
                b.read(&mtp.fc_embed, format!("{p}.fc_embed.weight"))?;
                b.read(&mtp.fc_hidden, format!("{p}.fc_hidden.weight"))?;
            } else {
                let half = extents(&mtp.fc_embed)[1];
                let fc = format!("{p}.fc.weight");
                b.read_expr(&mtp.fc_embed, Expr::src(fc.clone()).slice(1, 0, half))?;
                b.read_expr(&mtp.fc_hidden, Expr::src(fc).slice(1, half, half))?;
            }

            let a = &mtp.attn;
            b.read(&mtp.mixer_norm, n("input_layernorm.weight"))?;
            b.read(&a.qg_proj, n("self_attn.q_proj.weight"))?;
            b.read(&a.k_proj, n("self_attn.k_proj.weight"))?;
            b.read(&a.v_proj, n("self_attn.v_proj.weight"))?;
            b.read(&a.o_proj, n("self_attn.o_proj.weight"))?;
            b.read(&a.q_norm, n("self_attn.q_norm.weight"))?;
            b.read(&a.k_norm, n("self_attn.k_norm.weight"))?;
            b.read(&mtp.mlp_norm, n("post_attention_layernorm.weight"))?;
            match &mtp.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?;
                    b.read(down, n("mlp.down_proj.weight"))?;
                }
                Mlp::Routed { .. } => {
                    return Err(Error::Illegible {
                        name: n("mlp"),
                        detail: "a draft head is one block and routes to no experts".to_string(),
                    });
                }
            }
            if let Some(norm) = &mtp.norm {
                b.read(norm, format!("{p}.norm.weight"))?;
            }
        }

        if let Some(dflash) = &self.dflash {
            dflash.bind_aux(&mut b, src, &norm)?;
        }

        Ok(b.build())
    }

    pub fn import_from_gguf(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        if self.tower.is_some() {
            return Err(Error::Illegible {
                name: "visual".to_string(),
                detail: "this SKU declares a vision tower and no GGUF spelling \
                         of one is settled; import it from the safetensors \
                         checkpoint"
                    .to_string(),
            });
        }
        if self.mtp.is_some() {
            return Err(Error::Illegible {
                name: "mtp".to_string(),
                detail: "this SKU declares an MTP draft head and no GGUF \
                         spelling of one is settled; import it from the \
                         safetensors checkpoint"
                    .to_string(),
            });
        }
        let minus_one = |read: Expr| -> Expr { read.bias(-1.0) };

        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, "token_embd.weight")?;
        b.read_over(&self.final_norm, "output_norm.weight", minus_one)?;

        if let Head::Bank(head) = &self.head {
            b.read(
                head,
                spelled(
                    src,
                    ["output.weight".to_string(), "token_embd.weight".to_string()],
                )?,
            )?;
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");

            b.read_over(&w.mixer_norm, n("attn_norm.weight"), minus_one)?;
            b.read_over(
                &w.mlp_norm,
                spelled(src, [n("ffn_norm.weight"), n("post_attention_norm.weight")])?,
                minus_one,
            )?;

            match &w.mixer {
                Mixer::Attn(a) => {
                    b.read(&a.qg_proj, n("attn_q.weight"))?;
                    b.read(&a.k_proj, n("attn_k.weight"))?;
                    b.read(&a.v_proj, n("attn_v.weight"))?;
                    b.read(&a.o_proj, n("attn_output.weight"))?;
                    b.read_over(&a.q_norm, n("attn_q_norm.weight"), minus_one)?;
                    b.read_over(&a.k_norm, n("attn_k_norm.weight"), minus_one)?;
                }
                Mixer::Gdn(g) => {
                    match spelled(src, [n("ssm_in.weight")]) {
                        Ok(fused) => b.read(&g.in_qkvz, fused)?,
                        Err(_) => b.read_concat(
                            &g.in_qkvz,
                            [n("attn_qkv.weight"), n("attn_gate.weight")],
                        )?,
                    }
                    match spelled(src, [n("ssm_beta_alpha.weight")]) {
                        Ok(fused) => b.read(&g.in_ba, fused)?,
                        Err(_) => {
                            b.read_concat(&g.in_ba, [n("ssm_beta.weight"), n("ssm_alpha.weight")])?
                        }
                    }
                    b.read(&g.conv, n("ssm_conv1d.weight"))?;
                    b.read(&g.dt_bias, n("ssm_dt.bias"))?;
                    match spelled(src, [n("ssm_a_log")]) {
                        Ok(logarithm) => b.read(&g.a_log, logarithm)?,
                        Err(_) if src.get(&n("ssm_a")).is_some() => {
                            b.read_expr(&g.a_log, Expr::Src(n("ssm_a")).unary(UnaryOp::NegLn))?;
                        }
                        Err(missing) => return Err(missing),
                    }
                    b.read(&g.norm, n("ssm_norm.weight"))?;
                    b.read(&g.out_proj, n("ssm_out.weight"))?;
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(gate_up, [n("ffn_gate.weight"), n("ffn_up.weight")])?;
                    b.read(down, n("ffn_down.weight"))?;
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
                    b.read(router, n("ffn_gate_inp.weight"))?;

                    b.read_concat(
                        gate_up,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?;
                    b.read(down, n("ffn_down_exps.weight"))?;
                    b.read_concat(
                        shared_gate_up,
                        [n("ffn_gate_shexp.weight"), n("ffn_up_shexp.weight")],
                    )?;
                    b.read(shared_down, n("ffn_down_shexp.weight"))?;
                    b.read(shared_gate, n("ffn_gate_inp_shexp.weight"))?;
                }
            }
        }

        Ok(b.build())
    }
}

pub(crate) fn spelled<const N: usize>(
    src: &ztensor::Source,
    names: [String; N],
) -> Result<String, Error> {
    match names.iter().find(|name| src.get(name).is_some()) {
        Some(found) => Ok(found.clone()),
        None => Err(Error::Missing(names.join("` or `"))),
    }
}

pub(crate) fn flattened(
    src: &ztensor::Source,
    from: String,
    want: Vec<i64>,
) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let stored: i128 = shape.iter().map(|&n| i128::from(n)).product();
    let asked: i128 = want.iter().map(|&n| i128::from(n)).product();
    if stored > 1 && stored != asked {
        return Err(illegible(&format!(
            "is stored {shape:?} ({stored} elements) and the plan reads it as \
             {want:?} ({asked} elements)"
        )));
    }
    let encoding = checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(want, encoding)))
}

pub(crate) fn squeezed(src: &ztensor::Source, from: String) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let (channels, kernel) = match *shape {
        [channels, 1, kernel] => (channels, kernel),
        [channels, kernel, 1] => (channels, kernel),
        _ => {
            return Err(illegible(&format!(
                "a depthwise convolution bank is stored [channels, 1, kernel] \
                 or [channels, kernel, 1] and this one is stored {shape:?}"
            )));
        }
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
