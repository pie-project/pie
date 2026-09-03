use checkpoint::contract::{Expr, ModelContract, TensorType};

use super::model::{Head, Mixer, Mlp, Model};
use model_dsl::Platform;
use checkpoint_dsl::{Builder, Error, extents};

/// Where a safetensors checkpoint of this family puts its trunk. Two
/// spellings: transformers ships `model.language_model.layers.*` with a bare
/// `lm_head.weight`; `mlx_lm.convert` re-roots it under
/// `language_model.model.layers.*` / `language_model.lm_head.weight`. Every
/// leaf below `layers.{l}.` is spelled the same in both.
#[derive(Clone, Copy)]
enum Layout {
    /// `model.language_model.*` + `lm_head.weight` — transformers.
    Transformers,
    /// `language_model.model.*` + `language_model.lm_head.weight` — `mlx_lm`.
    Mlx,
}

impl Layout {
    /// What to call this layout in a refusal an operator reads.
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

    /// The tower is renamed, not just re-rooted: transformers publishes
    /// `model.visual.*`, `mlx_lm` publishes `vision_tower.*`. Leaf sets below
    /// that are otherwise identical.
    fn tower(self, leaf: &str) -> String {
        match self {
            Self::Transformers => format!("model.visual.{leaf}"),
            Self::Mlx => format!("vision_tower.{leaf}"),
        }
    }

    /// This family's RMSNorm computes `x_norm * (1 + w)`. `mlx_lm`'s
    /// `sanitize` folds the `+1` into the stored weight on conversion, so
    /// this reader subtracts it back out for every plain-RMSNorm plane.
    /// `Qwen3_5RMSNormGated` (`linear_attn.norm.weight`) has no constant and
    /// is not folded.
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
        // Each layout is picked by attempting to build it, not by sniffing a
        // witness tensor name: a mixed checkpoint can carry some planes
        // under one layout's names and the rest under another's.
        let mut refusals: Vec<String> = Vec::new();
        let mut attempt = |what: &str, built: Result<ModelContract, Error>| match built {
            Ok(contract) => Some(contract),
            Err(why) => {
                refusals.push(format!("as {what}, {why}"));
                None
            }
        };
        for layout in [Layout::Transformers, Layout::Mlx] {
            if let Some(contract) =
                attempt(layout.spelling(), self.import_from_safetensors(src, platform, layout))
            {
                return Ok(contract);
            }
        }
        if let Some(contract) = attempt("gguf", self.import_from_gguf(src, platform)) {
            return Ok(contract);
        }
        Err(Error::Illegible {
            name: "qwen_3".to_string(),
            // Each arm's own refusal, naming the first plane it wanted.
            detail: format!(
                "no reading of this file lands every plane this family \
                 declares — {}",
                refusals.join("; "),
            ),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source, platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from_safetensors(src, platform, Layout::Transformers)
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source, platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        // Undoes mlx_lm's RMSNorm +1 fold where the layout carries it; see
        // [`Layout::folds_the_norm_one`].
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
                        [n("linear_attn.in_proj_qkv.weight"), n("linear_attn.in_proj_z.weight")],
                    )?;

                    b.read_concat(
                        &g.in_ba,
                        [n("linear_attn.in_proj_b.weight"), n("linear_attn.in_proj_a.weight")],
                    )?;

                    b.read_expr(&g.conv, (|| -> Result<Expr, Error> {
                        squeezed(src, n("linear_attn.conv1d.weight"))
                    })()?)?;

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

                    // transformers ships gate_up fused as one
                    // `mlp.experts.gate_up_proj`; mlx_lm splits it into
                    // `switch_mlp.gate_proj` / `switch_mlp.up_proj`, joined
                    // here on the weight's own cut axis. `down_proj` is one
                    // tensor in both, differing only in name.
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

        // `patch_embed.proj.weight` is a Conv3d kernel read as a matmul bank
        // `[hidden, C*T*P^2]`: a torch (transformers) conv is already stored
        // in that byte order (a transmute); an MLX conv is channels-last, so
        // its rows need a permutation, done below.
        if let Some(t) = &self.tower {
            let v = |s: &str| layout.tower(s);
            const CHANNELS: i64 = 3;
            let want = extents(&t.patch_embed);
            b.read_expr(&t.patch_embed, (|| -> Result<Expr, Error> {
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
            })()?)?;
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

        // An overlay head ([`Recipe::Eagle`](super::model::Recipe::Eagle)) is
        // copied in by `--aux`, which is family-blind: it prefixes every
        // tensor name `aux.` and this block names them by the family's own
        // block spelling.
        if let Some(mtp) = &self.mtp {
            // A head published on its own (`mlx-community/*-MTP-4bit` is the
            // head alone, `fc.*` and `layers.0.*` at its root) rides in by
            // `--aux`, which prefixes it `aux.`; a checkpoint that carries
            // the head names it by the recipe's own prefix.
            let p: String = if src.get("aux.fc.weight").is_some() {
                "aux".to_string()
            } else {
                mtp.recipe.prefix().to_string()
            };
            let n = |s: &str| format!("{p}.layers.0.{s}");
            if let Some(pre) = &mtp.pre_fc {
                b.read(&pre.embedding, format!("{p}.pre_fc_norm_embedding.weight"))?;
                b.read(&pre.hidden, format!("{p}.pre_fc_norm_hidden.weight"))?;
            }

            // `mtp.fc.weight` is `[hidden, 2*hidden]`; columns `0..hidden`
            // are the embedding half, `hidden..2*hidden` the hidden half.
            let half = extents(&mtp.fc_embed)[1];
            let fc = format!("{p}.fc.weight");
            b.read_expr(&mtp.fc_embed, Expr::src(fc.clone()).slice(1, 0, half))?;
            b.read_expr(&mtp.fc_hidden, Expr::src(fc).slice(1, half, half))?;

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
                    b.read_concat(gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
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

        Ok(b.build())
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source, platform: Platform) -> Result<ModelContract, Error> {
        // GGUF has no settled spelling for a vision tower or a draft head;
        // refuse explicitly rather than publish a contract that silently
        // half-loads.
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
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, "token_embd.weight")?;
        b.read(&self.final_norm, "output_norm.weight")?;

        if let Head::Bank(head) = &self.head {
            b.read(head, "output.weight")?;
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");

            b.read(&w.mixer_norm, n("attn_norm.weight"))?;
            b.read(&w.mlp_norm, n("ffn_norm.weight"))?;

            match &w.mixer {
                Mixer::Attn(a) => {
                    b.read(&a.qg_proj, n("attn_q.weight"))?;
                    b.read(&a.k_proj, n("attn_k.weight"))?;
                    b.read(&a.v_proj, n("attn_v.weight"))?;
                    b.read(&a.o_proj, n("attn_output.weight"))?;
                    b.read(&a.q_norm, n("attn_q_norm.weight"))?;
                    b.read(&a.k_norm, n("attn_k_norm.weight"))?;
                }
                Mixer::Gdn(g) => {
                    b.read(&g.in_qkvz, n("ssm_in.weight"))?;
                    b.read(&g.in_ba, n("ssm_beta_alpha.weight"))?;
                    b.read(&g.conv, n("ssm_conv1d.weight"))?;
                    b.read(&g.dt_bias, n("ssm_dt.bias"))?;
                    b.read(&g.a_log, n("ssm_a"))?;
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

                    b.read_concat(gate_up, [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")])?;
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

/// The same tensor, re-typed to a stated shape — a `Conv3d` kernel read as the
/// matmul bank it already is. `[hidden, C, T, P, P]` and `[hidden, C*T*P^2]`
/// are the same bytes in the same order, so this is a transmute: it checks
/// the element count and re-states the type.
pub(crate) fn flattened(src: &ztensor::Source, from: String, want: Vec<i64>) -> Result<Expr, Error> {
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
    // Skipped when stored == 1: a source that states names only (no extents)
    // reports every shape as [1], which is not a real mismatch.
    if stored > 1 && stored != asked {
        return Err(illegible(&format!(
            "is stored {shape:?} ({stored} elements) and the plan reads it as \
             {want:?} ({asked} elements)"
        )));
    }
    let encoding = checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(want, encoding)))
}

/// `pub(crate)` for `qwen_4`, whose depthwise banks — the GDN convolutions
/// and the PLE's dilated one — are published in the same two spellings.
pub(crate) fn squeezed(src: &ztensor::Source, from: String) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    // Transformers stores a depthwise conv1d [channels, in/groups, kernel];
    // MLX stores [channels, kernel, in/groups]. in/groups is 1 for a
    // depthwise bank, so the two differ only in which axis carries the 1.
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
