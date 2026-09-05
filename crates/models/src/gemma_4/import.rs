use checkpoint::contract::{Expr, ModelContract};

use super::model::{AttnBanks, Model};
use checkpoint::contract::TensorType;

use model_dsl::Platform;
use checkpoint_dsl::{Builder, Error, extents};

/// Where a safetensors checkpoint puts its trunk: transformers and `mlx_lm`
/// spell the same tensors under different path prefixes; everything below
/// `layers.{l}.` is identical either way.
///
/// Unlike gemma 3, gemma 4's RMSNorm has no `+1` offset — checkpoint norm
/// weights are plain multiplicative scales.
#[derive(Clone, Copy)]
enum Layout {
    /// `model.language_model.*` — transformers.
    Transformers,
    /// `language_model.model.*` — `mlx_lm`.
    Mlx,
}

impl Layout {
    /// The trunk prefix, up to and including the trailing dot.
    fn trunk(self) -> &'static str {
        match self {
            Self::Transformers => "model.language_model.",
            Self::Mlx => "language_model.model.",
        }
    }

    fn at(self, leaf: &str) -> String {
        format!("{}{leaf}", self.trunk())
    }

    fn layer(self, l: usize, leaf: &str) -> String {
        format!("{}layers.{l}.{leaf}", self.trunk())
    }

    fn embed(self) -> String {
        self.at("embed_tokens.weight")
    }

    /// mlx_lm strips the `model.` prefix here too but does not relocate it,
    /// unlike the trunk.
    fn vision(self, leaf: &str) -> String {
        match self {
            Self::Transformers => format!("model.vision_tower.{leaf}"),
            Self::Mlx => format!("vision_tower.{leaf}"),
        }
    }

    /// The multimodal embedder's projection, which lives beside the tower and
    /// not under it.
    fn embed_vision(self) -> &'static str {
        match self {
            Self::Transformers => "model.embed_vision.embedding_projection.weight",
            Self::Mlx => "embed_vision.embedding_projection.weight",
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        // Try the native contract first: a file this crate's own import
        // wrote satisfies it directly, with no transform.
        // Otherwise try each safetensors layout, then GGUF, chosen by
        // building the contract rather than sniffing a name.
        let mut refusals: Vec<String> = Vec::new();
        for (what, layout) in [
            ("transformers", Layout::Transformers),
            ("mlx_lm", Layout::Mlx),
        ] {
            match self.import_from_safetensors(src, platform, layout) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {what}, {why}")),
            }
        }
        match self.import_from_gguf(src, platform) {
            Ok(contract) => return Ok(contract),
            Err(why) => refusals.push(format!("as gguf, {why}")),
        }
        Err(Error::Illegible {
            name: "gemma4".to_string(),
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
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, layout.embed())?;
        b.read(&self.final_norm, layout.at("norm.weight"))?;

        for (l, w) in self.layers.iter().enumerate() {
            let n = |leaf: &str| layout.layer(l, leaf);

            b.read(&w.attn_norm, n("input_layernorm.weight"))?;
            b.read(&w.post_attn_norm, n("post_attention_layernorm.weight"))?;
            b.read(&w.pre_ffw_norm, n("pre_feedforward_layernorm.weight"))?;
            b.read(&w.post_ffw_norm, n("post_feedforward_layernorm.weight"))?;
            b.read(&w.attn.q_norm, n("self_attn.q_norm.weight"))?;
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    b.read(k_norm, n("self_attn.k_norm.weight"))?;
                    let k = n("self_attn.k_proj.weight");
                    let v = n("self_attn.v_proj.weight");
                    // `attention_k_eq_v` layers publish no `v_proj`; the
                    // value leg then reads the key projection's own bytes.
                    let value = if src.get(&v).is_some() { v } else { k.clone() };
                    b.read_concat(qkv, [n("self_attn.q_proj.weight"), k, value])?;
                }

                AttnBanks::Shared { q_proj } => {
                    b.read(q_proj, n("self_attn.q_proj.weight"))?;
                }
            }
            b.read(&w.o_proj, n("self_attn.o_proj.weight"))?;
            b.read_concat(&w.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
            b.read(&w.down, n("mlp.down_proj.weight"))?;

            // The routed branch, where the checkpoint ships one. The router
            // norm's gain is the stored `router.scale * hidden**-0.5`, folded
            // into the plane rather than the forward. Expert gate/up banks
            // are stored split (`switch_glu.gate_proj` / `.up_proj`), not
            // fused, and `read_concat` joins them.
            if let Some(x) = &w.moe {
                let root = (self.hidden as f32).powf(-0.5);
                b.read_expr(
                    &x.router_norm,
                    Expr::src(n("router.scale")).scale(root),
                )?;
                b.read(&x.router, n("router.proj.weight"))?;
                b.read(&x.per_expert_scale, n("router.per_expert_scale"))?;
                b.read(&x.pre_ffw_norm_2, n("pre_feedforward_layernorm_2.weight"))?;
                b.read(&x.post_ffw_norm_1, n("post_feedforward_layernorm_1.weight"))?;
                b.read(&x.post_ffw_norm_2, n("post_feedforward_layernorm_2.weight"))?;
                b.read_concat(
                    &x.gate_up,
                    [
                        n("experts.switch_glu.gate_proj.weight"),
                        n("experts.switch_glu.up_proj.weight"),
                    ],
                )?;
                b.read(&x.down, n("experts.switch_glu.down_proj.weight"))?;
            }

            // A layer without PLE still owns its own `layer_scalar`; a PLE
            // stack's is read below instead.
            if let Some(scalar) = &w.scalar {
                b.read(scalar, n("layer_scalar"))?;
            }
        }

        if let Some(ple) = &self.ple {
            // The leading rows the declaration states: the whole plane for
            // the full stack, its first `layers * dim` rows for a miniature
            // cut below the checkpoint's depth.
            let rows = i64::try_from(ple.model_proj.shape[0]).expect("a row count inside i64");
            let name = layout.at("per_layer_model_projection.weight");
            let stored = src.get(&name).and_then(|t| t.shape().first().copied());
            if stored.is_some_and(|stored| i64::try_from(stored).ok() != Some(rows)) {
                b.read_expr(&ple.model_proj, Expr::src(name).slice(0, 0, rows))?;
            } else {
                b.read(&ple.model_proj, name)?;
            }
            b.read(&ple.model_norm, layout.at("per_layer_projection_norm.weight"))?;
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let n = |leaf: &str| layout.layer(l, leaf);
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                b.read_expr(
                    &p.table,
                    Expr::src(layout.at("embed_tokens_per_layer.weight")).slice(1, at, width),
                )?;
                b.read(&p.gate, n("per_layer_input_gate.weight"))?;
                b.read(&p.proj, n("per_layer_projection.weight"))?;
                b.read(&p.norm, n("post_per_layer_input_norm.weight"))?;

                b.read(&p.scalar, n("layer_scalar"))?;
            }
        }

        // Every tower plane is a plain read except the position table:
        // stored `[2, positions, hidden]`, read as one `[2 * positions,
        // hidden]` bank (contiguous transmute) for a two-tap
        // `embed_weighted`.
        if let Some(t) = &self.tower {
            let v = |s: &str| layout.vision(s);
            b.read(&t.patch_embed, v("patch_embedder.input_proj.weight"))?;
            b.read_expr(&t.pos_embed, (|| -> Result<Expr, Error> {
                flattened(
                    src,
                    v("patch_embedder.position_embedding_table"),
                    extents(&t.pos_embed),
                )
            })()?)?;
            b.read(&t.projection, layout.embed_vision())?;
            // Applied as `(h - std_bias) * std_scale`, when the tower states one.
            if let Some(std) = &t.std {
                b.read(&std.bias, v("std_bias"))?;
                b.read(&std.scale, v("std_scale"))?;
            }
            for (l, blk) in t.blocks.iter().enumerate() {
                let n = |s: &str| v(&format!("encoder.layers.{l}.{s}"));
                for (weight, from) in [
                    (&blk.attn_norm, n("input_layernorm.weight")),
                    (&blk.post_attn_norm, n("post_attention_layernorm.weight")),
                    (&blk.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
                    (&blk.post_ffw_norm, n("post_feedforward_layernorm.weight")),
                    (&blk.q_norm, n("self_attn.q_norm.weight")),
                    (&blk.k_norm, n("self_attn.k_norm.weight")),
                ] {
                    b.read(weight, from)?;
                }
                for (c, stem) in [
                    (&blk.q, n("self_attn.q_proj")),
                    (&blk.k, n("self_attn.k_proj")),
                    (&blk.v, n("self_attn.v_proj")),
                    (&blk.o, n("self_attn.o_proj")),
                    (&blk.gate, n("mlp.gate_proj")),
                    (&blk.up, n("mlp.up_proj")),
                    (&blk.down, n("mlp.down_proj")),
                ] {
                    // The bank always sits under `.linear.`; the clip flag
                    // only controls whether the four bound scalars are present.
                    b.read(&c.bank, format!("{stem}.linear.weight"))?;
                    // The four bounds. Stored as rank-0 scalars, read as `[1]`.
                    if let Some(k) = &c.clip {
                        for (weight, suffix) in [
                            (&k.in_lo, "input_min"),
                            (&k.in_hi, "input_max"),
                            (&k.out_lo, "output_min"),
                            (&k.out_hi, "output_max"),
                        ] {
                            b.read_expr(weight, (|| -> Result<Expr, Error> {
                                flattened(src, format!("{stem}.{suffix}"), extents(weight))
                            })()?)?;
                        }
                    }
                }
            }
        }

        // The aux draft head: `pie model import --aux` prefixes a second
        // checkpoint's names with `aux.`.
        if let Some(a) = &self.draft {
            // `aux.fc.weight` is `[hidden, 2*hidden]`, embedding half first.
            let half = extents(&a.fc_embed)[1];
            b.read_expr(&a.fc_embed, Expr::src("aux.fc.weight").slice(1, 0, half))?;
            b.read_expr(&a.fc_hidden, Expr::src("aux.fc.weight").slice(1, half, half))?;
            let n = |s: &str| format!("aux.layers.0.{s}");
            for (weight, from) in [
                (&a.attn_norm, n("input_layernorm.weight")),
                (&a.post_attn_norm, n("post_attention_layernorm.weight")),
                (&a.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
                (&a.post_ffw_norm, n("post_feedforward_layernorm.weight")),
                (&a.attn.q_norm, n("self_attn.q_norm.weight")),
                (&a.o_proj, n("self_attn.o_proj.weight")),
            ] {
                b.read(weight, from)?;
            }
            if let AttnBanks::Owned { qkv, k_norm, .. } = &a.attn.banks {
                b.read_concat(
                    qkv,
                    [
                        n("self_attn.q_proj.weight"),
                        n("self_attn.k_proj.weight"),
                        n("self_attn.v_proj.weight"),
                    ],
                )?;
                b.read(k_norm, n("self_attn.k_norm.weight"))?;
            }
            b.read_concat(&a.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
            b.read(&a.down, n("mlp.down_proj.weight"))?;
        }

        // Google's assistant, under the same `aux.` prefix, in its own
        // (transformers) spelling.
        if let Some(a) = &self.assistant {
            // `pre_projection` is one `[hidden, 2 * trunk]` bank in the
            // published checkpoint, sliced here; a restatement that split
            // it first (`scripts/quantize_assistant.py`, whose halves are
            // quantized on their own) names the halves.
            if src.get("aux.pre_projection_embed.weight").is_some() {
                b.read(&a.pre_embed, "aux.pre_projection_embed.weight")?;
                b.read(&a.pre_hidden, "aux.pre_projection_hidden.weight")?;
            } else {
                let th = extents(&a.pre_embed)[1];
                b.read_expr(&a.pre_embed, Expr::src("aux.pre_projection.weight").slice(1, 0, th))?;
                b.read_expr(&a.pre_hidden, Expr::src("aux.pre_projection.weight").slice(1, th, th))?;
            }
            b.read(&a.post, "aux.post_projection.weight")?;
            b.read(&a.embed, "aux.model.embed_tokens.weight")?;
            b.read(&a.norm, "aux.model.norm.weight")?;
            for (l, w) in a.layers.iter().enumerate() {
                let n = |s: &str| format!("aux.model.layers.{l}.{s}");
                let AttnBanks::Shared { q_proj } = &w.attn.banks else {
                    unreachable!("the assistant's attention is declared shared");
                };
                for (weight, from) in [
                    (&w.attn_norm, n("input_layernorm.weight")),
                    (&w.post_attn_norm, n("post_attention_layernorm.weight")),
                    (&w.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
                    (&w.post_ffw_norm, n("post_feedforward_layernorm.weight")),
                    (&w.attn.q_norm, n("self_attn.q_norm.weight")),
                    (q_proj, n("self_attn.q_proj.weight")),
                    (&w.o_proj, n("self_attn.o_proj.weight")),
                    (&w.scalar, n("layer_scalar")),
                    (&w.down, n("mlp.down_proj.weight")),
                ] {
                    b.read(weight, from)?;
                }
                b.read_concat(&w.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
            }
        }

        // The block drafter, `--aux`-imported (it is published on its own);
        // its planes and their spelling are the drafter's (`drafter::dflash`).
        // Its norms are a Qwen3-style stack's, not gemma's `1 + w`, and the
        // drafter's text reads them through the `+1` op, so the stored weight
        // is read down by one — the fold the qwen families measured.
        if let Some(dflash) = &self.dflash {
            dflash.bind_aux(&mut b, src, &|from| Expr::src(from).bias(-1.0))?;
        }

        Ok(b.build())
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source, platform: Platform) -> Result<ModelContract, Error> {
        if self.draft.is_some() || self.assistant.is_some() {
            return Err(Error::Illegible {
                name: "aux".to_string(),
                detail: "this SKU declares an aux draft head and no GGUF \
                         spelling of one is settled; import it from the \
                         safetensors artifact"
                    .to_string(),
            });
        }
        if self.tower.is_some() {
            return Err(Error::Illegible {
                name: "vision_tower".to_string(),
                detail: "this SKU declares a vision tower and no GGUF spelling \
                         of one is settled; import it from the safetensors \
                         checkpoint"
                    .to_string(),
            });
        }
        if self.layers.iter().any(|w| w.moe.is_some()) {
            return Err(Error::Illegible {
                name: "experts".to_string(),
                detail: "this SKU declares a routed feedforward branch and no \
                         GGUF spelling of gemma 4's `experts.switch_glu.*` or \
                         `router.*` is settled; import it from the \
                         safetensors checkpoint"
                    .to_string(),
            });
        }
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, "token_embd.weight")?;
        b.read(&self.final_norm, "output_norm.weight")?;

        for (l, w) in self.layers.iter().enumerate() {
            b.read(&w.attn_norm, format!("blk.{l}.attn_norm.weight"))?;
            b.read(&w.post_attn_norm, format!("blk.{l}.post_attention_norm.weight"))?;
            b.read(&w.pre_ffw_norm, format!("blk.{l}.ffn_norm.weight"))?;
            b.read(&w.post_ffw_norm, format!("blk.{l}.post_ffw_norm.weight"))?;
            b.read(&w.attn.q_norm, format!("blk.{l}.attn_q_norm.weight"))?;
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    b.read(k_norm, format!("blk.{l}.attn_k_norm.weight"))?;
                    b.read_concat(
                        qkv,
                        [
                            format!("blk.{l}.attn_q.weight"),
                            format!("blk.{l}.attn_k.weight"),
                            format!("blk.{l}.attn_v.weight"),
                        ],
                    )?;
                }

                AttnBanks::Shared { q_proj } => {
                    b.read(q_proj, format!("blk.{l}.attn_q.weight"))?;
                }
            }
            b.read(&w.o_proj, format!("blk.{l}.attn_output.weight"))?;
            b.read_concat(
                &w.gate_up,
                [format!("blk.{l}.ffn_gate.weight"), format!("blk.{l}.ffn_up.weight")],
            )?;
            b.read(&w.down, format!("blk.{l}.ffn_down.weight"))?;
            // A layer without PLE owns its own scalar; a PLE stack's is read below.
            if let Some(scalar) = &w.scalar {
                b.read(scalar, format!("blk.{l}.layer_scalar"))?;
            }
        }

        if let Some(ple) = &self.ple {
            b.read(&ple.model_proj, "per_layer_model_proj.weight")?;
            b.read(&ple.model_norm, "per_layer_proj_norm.weight")?;
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                b.read_expr(
                    &p.table,
                    Expr::src("per_layer_token_embd.weight").slice(1, at, width),
                )?;
                b.read(&p.gate, format!("blk.{l}.inp_gate.weight"))?;
                b.read(&p.proj, format!("blk.{l}.proj.weight"))?;
                b.read(&p.norm, format!("blk.{l}.post_norm.weight"))?;

                b.read(&p.scalar, format!("blk.{l}.layer_scalar"))?;
            }
        }

        Ok(b.build())
    }
}

/// Re-types a tensor to a stated shape without moving bytes: checks the
/// element count matches, then transmutes. A source with no stated extents
/// is passed through unchecked (the plan compiler checks it later).
fn flattened(src: &ztensor::Source, from: String, want: Vec<i64>) -> Result<Expr, Error> {
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
