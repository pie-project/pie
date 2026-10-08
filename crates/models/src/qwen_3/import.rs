use checkpoint::contract::{Expr, ModelContract, TensorType, UnaryOp};

use super::model::{Head, Mixer, Mlp, Model};
use super::rotation;
use checkpoint_dsl::format::{Format, Stated, attribute, attribute_text, config, has, read_one};
use checkpoint_dsl::{Builder, Error, extents};
use poem_dsl::Platform;

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
        let arch = attribute_text(src, "general.architecture")
            .unwrap_or_default()
            .to_string();
        read_one(
            "qwen_3",
            src,
            vec![
                Format::new(
                    Layout::Transformers.spelling(),
                    |src| has(src, Layout::Transformers.embed()),
                    || self.import_from_safetensors(src, platform, Layout::Transformers),
                )
                .stating(self.configured("text_config.")),
                Format::new(
                    Layout::Mlx.spelling(),
                    |src| has(src, Layout::Mlx.embed()),
                    || self.import_from_safetensors(src, platform, Layout::Mlx),
                )
                .stating(self.configured("text_config.")),
                Format::new(
                    "gguf",
                    |src| attribute_text(src, "general.architecture").is_some(),
                    || self.import_from_gguf(src, platform),
                )
                .stating(self.attributed(&arch)),
            ],
        )
    }

    /// The shape a transformers configuration states of this model, its text
    /// model's keys under `at`.
    fn configured(&self, at: &str) -> Vec<Stated> {
        let mut states = vec![
            config(format!("{at}hidden_size"), self.hidden),
            config(format!("{at}vocab_size"), self.vocab),
            config(format!("{at}num_attention_heads"), self.q_heads),
            config(format!("{at}num_key_value_heads"), self.kv_heads),
            config(format!("{at}head_dim"), self.head_dim),
            config(format!("{at}num_hidden_layers"), self.layers.len() as u32).or_deeper(),
            config(format!("{at}rms_norm_eps"), self.final_norm_eps),
        ];
        if let Some(theta) = self.theta() {
            states.push(config(format!("{at}rope_parameters.rope_theta"), theta));
        }
        states
    }

    /// The shape a GGUF's metadata states of this model, under `arch`.
    fn attributed(&self, arch: &str) -> Vec<Stated> {
        let mut states = vec![
            attribute(format!("{arch}.embedding_length"), self.hidden),
            attribute(format!("{arch}.block_count"), self.layers.len() as u32).or_deeper(),
            attribute(format!("{arch}.attention.head_count"), self.q_heads),
            attribute(format!("{arch}.attention.head_count_kv"), self.kv_heads),
        ];
        if let Some(theta) = self.theta() {
            states.push(attribute(format!("{arch}.rope.freq_base"), theta));
        }
        states
    }

    fn theta(&self) -> Option<f32> {
        self.layers.iter().find_map(|layer| match &layer.mixer {
            Mixer::Attn(a) => Some(a.theta),
            Mixer::Gdn(_) => None,
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

        let mut b = Builder::new(src, 1, platform);
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

        // The Bonsai GGUF stores the GDN v-heads TILED ("v-grouped"); when the flag
        // is set the GDN reads below reorder every v-head-indexed tensor tiled→block
        // so pie's block-pairing scan kernel pairs v→k correctly. Absent/false => a
        // block-stored GGUF, left untouched.
        let v_grouped = rotation::gdn_v_grouped(src);

        let mut b = Builder::new(src, 1, platform);
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
                    // v-head geometry for the tiled→block σ reorder (gated on
                    // `v_grouped`). `prefix` = the q+k rows the fused qkv/conv carry
                    // ahead of the v-heads; `width` = rows per v-head.
                    let k_w = 2 * i64::from(g.k_heads) * i64::from(g.k_dim);
                    let heads = i64::from(g.v_heads);
                    let width = i64::from(g.v_dim);
                    let k_heads = i64::from(g.k_heads);
                    let rep = heads / k_heads;
                    let sigma = |prefix: i64, width: i64| {
                        v_head_reorder_rows(prefix, heads, width, k_heads, rep)
                    };

                    match spelled(src, [n("ssm_in.weight")]) {
                        // Fused `[q,k,v,z]`: σ the v block (at `k_w`) and the z block
                        // (at `k_w + v_heads*v_dim`), leaving q,k identity.
                        Ok(fused) if v_grouped => {
                            let z_base = k_w + heads * width;
                            let mut idx = sigma(k_w, width); // q,k identity ⧺ σ(v)
                            idx.extend(sigma(0, width).into_iter().map(|i| z_base + i)); // σ(z)
                            b.read_expr(&g.in_qkvz, Expr::src(fused).gather(0, idx))?;
                        }
                        Ok(fused) => b.read(&g.in_qkvz, fused)?,
                        // Split `attn_qkv` (q,k,v) ⧺ `attn_gate` (z): σ the v rows of
                        // qkv (after the `k_w` q/k prefix) and all of the gate.
                        Err(_) if v_grouped => {
                            let qkv = Expr::src(n("attn_qkv.weight")).gather(0, sigma(k_w, width));
                            let gate = Expr::src(n("attn_gate.weight")).gather(0, sigma(0, width));
                            b.read_expr(&g.in_qkvz, Expr::concat(0, vec![qkv, gate]))?;
                        }
                        Err(_) => b.read_concat(
                            &g.in_qkvz,
                            [n("attn_qkv.weight"), n("attn_gate.weight")],
                        )?,
                    }
                    // `in_ba` = `[ssm_beta, ssm_alpha]` (beta-first): the GGUF's
                    // `ssm_beta`/`ssm_alpha` are the fork's beta/alpha directly, and
                    // the GDN prep reads `in_ba` as `[beta, alpha]` — matching the HF
                    // path's `[in_proj_b, in_proj_a]`. Each is one row per v-head, so
                    // σ reorders those 48 rows when v-grouped. The leg order has one
                    // source of truth in `in_ba_legs` (Bug B shipped them alpha-first).
                    match spelled(src, [n("ssm_beta_alpha.weight")]) {
                        Ok(fused) if v_grouped => {
                            // Fused `[beta; alpha]` under v-grouping: σ-reorder each
                            // half (beta = rows 0..heads, alpha = rows heads..2·heads),
                            // mirroring the split legs' `sigma(0, 1)` and the qkvz path,
                            // so the gates pair with the right heads.
                            let mut idx = sigma(0, 1);
                            let heads = idx.len() as i64;
                            idx.extend(sigma(0, 1).into_iter().map(|i| heads + i));
                            b.read_expr(&g.in_ba, Expr::src(fused).gather(0, idx))?;
                        }
                        Ok(fused) => b.read(&g.in_ba, fused)?,
                        Err(_) if v_grouped => {
                            let [beta, alpha] = in_ba_legs(l);
                            let beta = Expr::src(beta).gather(0, sigma(0, 1));
                            let alpha = Expr::src(alpha).gather(0, sigma(0, 1));
                            b.read_expr(&g.in_ba, Expr::concat(0, vec![beta, alpha]))?;
                        }
                        Err(_) => b.read_concat(&g.in_ba, in_ba_legs(l))?,
                    }
                    if v_grouped {
                        // conv carries `[q,k,v]` channels; σ the v channels.
                        b.read_expr(
                            &g.conv,
                            Expr::src(n("ssm_conv1d.weight")).gather(0, sigma(k_w, width)),
                        )?;
                        b.read_expr(
                            &g.dt_bias,
                            Expr::src(n("ssm_dt.bias")).gather(0, sigma(0, 1)),
                        )?;
                    } else {
                        b.read(&g.conv, n("ssm_conv1d.weight"))?;
                        b.read(&g.dt_bias, n("ssm_dt.bias"))?;
                    }
                    match spelled(src, [n("ssm_a_log")]) {
                        Ok(logarithm) if v_grouped => {
                            b.read_expr(&g.a_log, Expr::src(logarithm).gather(0, sigma(0, 1)))?;
                        }
                        Ok(logarithm) => b.read(&g.a_log, logarithm)?,
                        Err(_) if src.get(&n("ssm_a")).is_some() => {
                            let a = Expr::src(n("ssm_a"));
                            let a = if v_grouped {
                                a.gather(0, sigma(0, 1))
                            } else {
                                a
                            };
                            b.read_expr(&g.a_log, a.unary(UnaryOp::NegLn))?;
                        }
                        Err(missing) => return Err(missing),
                    }
                    // `ssm_norm` is per-`v_dim` (shared across heads) and `ssm_out`
                    // is already block-canonical on its input axis (the fork reorders
                    // the ACTIVATION, not this weight) — neither carries a v-head axis
                    // to reorder.
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

/// Build a gather index that reorders a GDN tensor's v-head axis from the Bonsai
/// GGUF's **tiled** ("v-grouped") layout into the **block** layout pie's GDN scan
/// kernel pairs by (gated on [`rotation::GDN_V_GROUPED_KEY`]).
///
/// A leg has `prefix` leading rows that are NOT v-head indexed — the q/k rows of
/// the fused `qkv`/`conv` projection; `0` for the gate `z`, `ssm_beta`/`ssm_alpha`,
/// `dt_bias`, and `a_log` legs — then `heads` v-heads each `width` rows wide. In
/// the tiled file a v-head at block position `p` lives at tiled index
/// `(p % rep) * k_heads + p / rep` (`rep = heads / k_heads`); gather each block
/// position from there so the shared kernel's `p -> p / rep` v→k pairing lands the
/// k-head the tiled v-head truly belongs to (`((p % rep) * k_heads + p / rep) %
/// k_heads == p / rep`). The leg's output is then in block order end to end — the
/// scan emits block-ordered heads straight into the (already block-canonical)
/// `ssm_out`, so no output-side reorder is needed.
fn v_head_reorder_rows(prefix: i64, heads: i64, width: i64, k_heads: i64, rep: i64) -> Vec<i64> {
    let mut idx: Vec<i64> = (0..prefix).collect();
    for p in 0..heads {
        let old = (p % rep) * k_heads + (p / rep);
        let base = prefix + old * width;
        idx.extend(base..base + width);
    }
    idx
}

/// The GGUF source legs of a GDN layer's `in_ba`, **beta-first** —
/// `[ssm_beta, ssm_alpha]`. The GDN prep reads `in_ba` as `[beta, alpha]`
/// (matching the HF path's `[in_proj_b, in_proj_a]`), so the concat must land
/// beta in its first half. This is the one source of truth for that order; Bug B
/// shipped them alpha-first. [`tests::in_ba_concatenates_beta_first`] pins it.
fn in_ba_legs(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.ssm_beta.weight"),
        format!("blk.{layer}.ssm_alpha.weight"),
    ]
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

#[cfg(test)]
mod tests {
    use super::{in_ba_legs, v_head_reorder_rows};
    use crate::qwen_3::rotation::gdn_v_grouped_in;
    use ztensor::format::cbor::Value;

    /// The σ reorder lands every v-head where pie's block-pairing scan kernel
    /// (`v-head p -> k-head p / rep`) will pair it with the k-head the tiled file
    /// truly stored it against (`tiled index % k_heads`). This is the whole
    /// correctness property of Bug A's fix — verified over a synthetic layout,
    /// no GGUF needed.
    #[test]
    fn sigma_pairs_every_v_head_with_its_true_k_head() {
        // A tiny geometry mirroring the Bonsai ratio: k_heads=2, v_heads=6, rep=3,
        // one scalar per head (width 1), no q/k prefix.
        let (k_heads, heads, rep, width, prefix) = (2i64, 6i64, 3i64, 1i64, 0i64);
        let idx = v_head_reorder_rows(prefix, heads, width, k_heads, rep);
        assert_eq!(idx.len(), heads as usize);
        for p in 0..heads {
            let tiled = idx[p as usize];
            // the tiled file pairs v-head `t` with k-head `t % k_heads`; after the
            // reorder, block position `p` must land on that very k-head.
            assert_eq!(
                tiled % k_heads,
                p / rep,
                "block position {p} (k-head {}) drew tiled head {tiled} (k-head {})",
                p / rep,
                tiled % k_heads,
            );
        }
        // Exactly the fork's reshape{rep,k_heads}->transpose, spelled out.
        assert_eq!(idx, vec![0, 2, 4, 1, 3, 5]);
    }

    /// Multi-row heads keep each head's `width` rows contiguous and in order, and
    /// a non-v-head `prefix` (the fused q/k rows) passes through as identity.
    #[test]
    fn sigma_preserves_prefix_and_within_head_rows() {
        let (k_heads, heads, rep, width, prefix) = (2i64, 4i64, 2i64, 3i64, 5i64);
        let idx = v_head_reorder_rows(prefix, heads, width, k_heads, rep);
        // prefix rows are identity.
        assert_eq!(&idx[..prefix as usize], &[0, 1, 2, 3, 4]);
        // block head p draws tiled head old(p) = (p % rep) * k_heads + p / rep,
        // its `width` rows contiguous at prefix + old*width.
        for p in 0..heads {
            let old = (p % rep) * k_heads + (p / rep);
            let base = prefix + old * width;
            let at = (prefix + p * width) as usize;
            assert_eq!(
                &idx[at..at + width as usize],
                (base..base + width).collect::<Vec<_>>().as_slice(),
            );
        }
    }

    /// The reorder is gated: the flag must be present AND `true`. An absent map,
    /// an absent key, and a `false` value all leave a block-stored GGUF untouched.
    #[test]
    fn the_v_grouped_gate_demands_an_explicit_true() {
        let key = crate::qwen_3::rotation::GDN_V_GROUPED_KEY;
        let map = |b: bool| Value::Map(vec![(Value::Text(key.to_string()), Value::Bool(b))]);
        assert!(gdn_v_grouped_in(Some(&map(true))), "explicit true gates on");
        assert!(!gdn_v_grouped_in(Some(&map(false))), "false stays off");
        assert!(
            !gdn_v_grouped_in(Some(&Value::Map(vec![]))),
            "an absent key stays off"
        );
        assert!(!gdn_v_grouped_in(None), "no metadata map stays off");
    }

    /// Bug B guard — the `in_ba` concat lands BETA-FIRST. Build a tiny synthetic
    /// GGUF whose `ssm_beta` rows are all `1.0` and `ssm_alpha` rows all `2.0`
    /// (distinguishable), read it with the REAL ztensor GGUF reader, run the
    /// import's own `read_concat(in_ba, in_ba_legs(..))` over it, materialise the
    /// `in_ba` plane through the REAL checkpoint executor, and assert its first
    /// half is beta and its second half is alpha. A beta/alpha swap in
    /// [`super::in_ba_legs`] flips these bytes and trips the test. No real GGUF,
    /// no Metal.
    #[test]
    fn in_ba_concatenates_beta_first() {
        use checkpoint::executor::Execution;
        use checkpoint::file::read::parse_metadata;
        use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
        use checkpoint_dsl::Builder;
        use poem_dsl::{Dtype, Platform, Weight};

        // Tiny GDN geometry: 2 v-heads, hidden 4. `in_ba` is `[2*v_heads, hidden]`,
        // its first `v_heads` rows beta, its last `v_heads` rows alpha.
        const VH: u64 = 2;
        const HID: u64 = 4;
        const BETA: f32 = 1.0;
        const ALPHA: f32 = 2.0;
        const GGML_F32: u32 = 0;

        let fill =
            |val: f32| -> Vec<u8> { (0..VH * HID).flat_map(|_| val.to_le_bytes()).collect() };

        // A minimal GGUF v3 writer: `(name, logical shape, ggml type, payload)`.
        // The shape is written fastest-dim-first (ggml order), which the reader
        // reverses back to the logical `[rows, cols]`.
        fn gguf(tensors: &[(&str, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
            const ALIGN: usize = 32;
            let mut head = Vec::new();
            head.extend_from_slice(b"GGUF");
            head.extend_from_slice(&3u32.to_le_bytes());
            head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            head.extend_from_slice(&0u64.to_le_bytes()); // kv_count
            let mut data = Vec::new();
            for (name, shape, type_id, payload) in tensors {
                head.extend_from_slice(&(name.len() as u64).to_le_bytes());
                head.extend_from_slice(name.as_bytes());
                head.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                for dim in shape.iter().rev() {
                    head.extend_from_slice(&dim.to_le_bytes());
                }
                head.extend_from_slice(&type_id.to_le_bytes());
                head.extend_from_slice(&(data.len() as u64).to_le_bytes());
                data.extend_from_slice(payload);
                while !data.len().is_multiple_of(ALIGN) {
                    data.push(0);
                }
            }
            while !head.len().is_multiple_of(ALIGN) {
                head.push(0);
            }
            head.extend_from_slice(&data);
            head
        }

        let bytes = gguf(&[
            ("blk.0.ssm_beta.weight", vec![VH, HID], GGML_F32, fill(BETA)),
            (
                "blk.0.ssm_alpha.weight",
                vec![VH, HID],
                GGML_F32,
                fill(ALPHA),
            ),
        ]);
        let dir = std::env::temp_dir().join(format!("bonsai_in_ba_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("in_ba.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let src =
            ztensor_compat::index(&path).expect("the real gguf reader opens the synthetic file");
        let metadata = parse_metadata(&path).expect("parse the synthetic GGUF metadata");

        // The import's own in_ba read: `[ssm_beta, ssm_alpha]` via `in_ba_legs`,
        // declared exactly as `d27b_bonsai`'s GDN `in_ba` (bf16, two v-head legs).
        let in_ba = Weight::sym("in_ba", [2 * VH, HID], Dtype::Bf16).packed([VH, VH]);
        let mut b = Builder::new(&src, 1, Platform::Metal);
        b.read_concat(&in_ba, in_ba_legs(0))
            .expect("read the in_ba concat");
        let contract = b.build();

        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = checkpoint::plan::compile(&metadata, &contract, target)
            .expect("compile the in_ba read");
        let storage = Execution::new(&plan, &dir)
            .run()
            .expect("materialise in_ba");
        let raw = storage
            .tensors
            .get("in_ba")
            .expect("the plan materialises `in_ba`");

        // Decode the bf16 plane (`±` exact) and split beta-half / alpha-half.
        let vals: Vec<f32> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
            .collect();
        assert_eq!(
            vals.len(),
            (2 * VH * HID) as usize,
            "in_ba is [2*v_heads, hidden]"
        );
        let (first, second) = vals.split_at((VH * HID) as usize);
        assert!(
            first.iter().all(|&v| v == BETA),
            "in_ba first half must be beta ({BETA}); got {first:?} — in_ba is alpha-first (Bug B)",
        );
        assert!(
            second.iter().all(|&v| v == ALPHA),
            "in_ba second half must be alpha ({ALPHA}); got {second:?}",
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
