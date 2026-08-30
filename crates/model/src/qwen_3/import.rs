use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};

use super::model::{Head, Mixer, Mlp, Model};
use crate::contract::{ALIGNMENT, ModelError, copy, declare, extents, fused, planes, planes_fused};

/// **WHERE A SAFETENSORS CHECKPOINT OF THIS FAMILY PUTS ITS TRUNK.** Two
/// spellings, and the difference is one swapped pair of path components.
///
/// Transformers publishes the multimodal wrapper first and the language tower
/// under it — `model.language_model.layers.*`, with the readout hoisted to a
/// bare `lm_head.weight` — and that is what `Qwen/Qwen3.5-0.8B` ships.
/// `mlx_lm.convert` re-roots the same tensors under its own module tree, whose
/// top level is the tower and whose `.model` is the decoder:
/// `language_model.model.layers.*`, `language_model.lm_head.weight`. Nothing
/// else moves — every leaf below `layers.{l}.` is spelled the same in both,
/// which is why this is a prefix and not a second import.
///
/// **THIS IS FIRST-LIGHT BREAKAGE, NOT A NEW FEATURE.** The MLX affine-U4 rows
/// were written against a description of what `mlx_lm` emits rather than
/// against an emitted file, and they named the transformers spelling. No
/// `mlx_lm` output has ever satisfied them: `import` refused at the door, on
/// the embedding, before one plane was read. `kimi_k3` already reads the MLX
/// spelling (`kimi_k3::import`'s `HF_EMBED`), so the tree knew both existed;
/// this family had only been told about one.
#[derive(Clone, Copy)]
enum Layout {
    /// `model.language_model.*` + `lm_head.weight` — transformers.
    Transformers,
    /// `language_model.model.*` + `language_model.lm_head.weight` — `mlx_lm`.
    Mlx,
}

impl Layout {
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

    /// **THE ONE THIS FAMILY'S RMSNORM ADDS, AND WHERE THE FILE ALREADY PUT
    /// IT.** `Qwen3_5RMSNorm` is `x_norm * (1 + w)` — transformers'
    /// `modeling_qwen3_5.py` says so in the operation and again in the
    /// initializer, which zeroes the parameter "to be 1 centered as the
    /// RMSNorm here does (1 + weight)" — and this tree's `RmsnormPlusOne`
    /// computes exactly that off the published `w`.
    ///
    /// **`mlx_lm` DOES NOT HAVE THAT KERNEL AND FOLDS THE ONE INTO THE
    /// WEIGHT INSTEAD.** Its `Qwen3_5Model.sanitize` adds `1.0` to every
    /// plane whose name ends in `.input_layernorm.weight`,
    /// `.post_attention_layernorm.weight`, `model.norm.weight`,
    /// `.q_norm.weight` or `.k_norm.weight`, so that a plain
    /// `nn.RMSNorm` — `x_norm * w`, no constant — reproduces the same model.
    /// `mlx_lm.convert` runs `sanitize` on the way in and writes the SHIFTED
    /// values out, so the fold is a permanent property of every MLX-layout
    /// artifact of this family, `mlx-community`'s included. Measured on a
    /// local conversion of `Qwen/Qwen3.5-0.8B`: `norm.weight` means +3.3092
    /// in the transformers file and +4.3084 in the MLX one, and
    /// `max|mlx - (hf + 1)|` is 0.0156 — bf16 rounding, and nothing else.
    ///
    /// **AN ENGINE THAT ADDS ITS OWN ONE TO A WEIGHT THAT ALREADY CARRIES ONE
    /// COMPUTES `x_norm * (2 + w)`**, which is finite, deterministic, and
    /// wrong at every norm in the stack. It was the whole of this SKU's first
    /// light: every bank landed byte-identical to the file and every affine
    /// point computed what a host reference said, and the model answered
    /// nonsense anyway.
    ///
    /// So the import takes the fold back out, which is what `Expr::Bias`
    /// exists for — its own doc names this exact disagreement, in gemma's
    /// spelling of it. **THE GATED NORM IS NOT IN THE LIST AND MUST NOT BE**:
    /// `Qwen3_5RMSNormGated` is `w * x_norm` with no constant, `sanitize`
    /// leaves `linear_attn.norm.weight` alone, and the two files hold it
    /// byte-identical.
    fn folds_the_norm_one(self) -> bool {
        match self {
            Self::Transformers => false,
            Self::Mlx => true,
        }
    }
}

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        let gguf = "token_embd.weight";
        for layout in [Layout::Transformers, Layout::Mlx] {
            if src.get(layout.embed()).is_some() {
                return self.import_from_safetensors(src, layout);
            }
        }
        if src.get(gguf).is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Illegible {
            name: "qwen_3".to_string(),
            detail: format!(
                "it holds none of `{}`, `{}` or `{gguf}`, so it is written \
                 in no format this family reads",
                Layout::Transformers.embed(),
                Layout::Mlx.embed(),
            ),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        self.import_from_safetensors(src, Layout::Transformers)
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source,
        layout: Layout,
    ) -> Result<ModelContract, ModelError> {
        // One plain-RMSNorm plane, with the fold `mlx_lm` baked in taken
        // back out where the file carries one — see
        // [`Layout::folds_the_norm_one`]. Every `copy` below that names a
        // NORM goes through this; the gated one does not, and the projections
        // never could.
        let norm = |w: &_, from: String| -> Result<TensorContract, ModelError> {
            let read = Expr::src(from);
            let read = if layout.folds_the_norm_one() {
                read.bias(-1.0)
            } else {
                read
            };
            declare(src, w, read)
        };

        let mut tensors = planes(src, &self.embed, layout.embed())?;
        tensors.push(norm(&self.final_norm, layout.norm().to_string())?);

        if let Head::Bank(head) = &self.head {
            tensors.extend(planes(src, head, layout.head())?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| layout.layer(l, s);

            tensors.push(norm(&w.mixer_norm, n("input_layernorm.weight"))?);
            tensors.push(norm(&w.mlp_norm, n("post_attention_layernorm.weight"))?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.extend(planes(src, &a.qg_proj, n("self_attn.q_proj.weight"))?);
                    tensors.extend(planes(src, &a.k_proj, n("self_attn.k_proj.weight"))?);
                    tensors.extend(planes(src, &a.v_proj, n("self_attn.v_proj.weight"))?);
                    tensors.extend(planes(src, &a.o_proj, n("self_attn.o_proj.weight"))?);
                    tensors.push(norm(&a.q_norm, n("self_attn.q_norm.weight"))?);
                    tensors.push(norm(&a.k_norm, n("self_attn.k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.extend(planes_fused(
                        src,
                        &g.in_qkvz,
                        [
                            n("linear_attn.in_proj_qkv.weight"),
                            n("linear_attn.in_proj_z.weight"),
                        ],
                    )?);

                    tensors.extend(planes_fused(
                        src,
                        &g.in_ba,
                        [
                            n("linear_attn.in_proj_b.weight"),
                            n("linear_attn.in_proj_a.weight"),
                        ],
                    )?);

                    tensors.push(declare(
                        src,
                        &g.conv,
                        squeezed(src, n("linear_attn.conv1d.weight"))?,
                    )?);

                    tensors.push(copy(src, &g.dt_bias, n("linear_attn.dt_bias"))?);
                    tensors.push(copy(src, &g.a_log, n("linear_attn.A_log"))?);
                    tensors.push(copy(src, &g.norm, n("linear_attn.norm.weight"))?);
                    tensors.extend(planes(src, &g.out_proj, n("linear_attn.out_proj.weight"))?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.extend(planes_fused(
                        src,
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?);
                    tensors.extend(planes(src, down, n("mlp.down_proj.weight"))?);
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
                    tensors.extend(planes(src, router, n("mlp.gate.weight"))?);

                    tensors.extend(planes(src, gate_up, n("mlp.experts.gate_up_proj"))?);
                    tensors.extend(planes(src, down, n("mlp.experts.down_proj"))?);
                    tensors.extend(planes_fused(
                        src,
                        shared_gate_up,
                        [
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    )?);
                    tensors.extend(planes(
                        src,
                        shared_down,
                        n("mlp.shared_expert.down_proj.weight"),
                    )?);
                    tensors.extend(planes(
                        src,
                        shared_gate,
                        n("mlp.shared_expert_gate.weight"),
                    )?);
                }
            }
        }

        // **THE DRAFT HEAD, VERIFIED BYTE-FOR-BYTE AGAINST THE CACHED
        // CHECKPOINT INDEX.** Fifteen tensors under `mtp.*` in
        // `models--Qwen--Qwen3.6-27B` at snapshot `6a9e13bd`, read out of
        // `model.safetensors.index.json` and the shard headers it names:
        //
        // ```text
        // mtp.fc.weight                                 BF16 [5120, 10240]
        // mtp.pre_fc_norm_embedding.weight              BF16 [5120]
        // mtp.pre_fc_norm_hidden.weight                 BF16 [5120]
        // mtp.layers.0.input_layernorm.weight           BF16 [5120]
        // mtp.layers.0.self_attn.q_proj.weight          BF16 [12288, 5120]
        // mtp.layers.0.self_attn.k_proj.weight          BF16 [1024, 5120]
        // mtp.layers.0.self_attn.v_proj.weight          BF16 [1024, 5120]
        // mtp.layers.0.self_attn.o_proj.weight          BF16 [5120, 6144]
        // mtp.layers.0.self_attn.q_norm.weight          BF16 [256]
        // mtp.layers.0.self_attn.k_norm.weight          BF16 [256]
        // mtp.layers.0.post_attention_layernorm.weight  BF16 [5120]
        // mtp.layers.0.mlp.gate_proj.weight             BF16 [17408, 5120]
        // mtp.layers.0.mlp.up_proj.weight               BF16 [17408, 5120]
        // mtp.layers.0.mlp.down_proj.weight             BF16 [5120, 17408]
        // mtp.norm.weight                               BF16 [5120]
        // ```
        //
        // Every attention and mlp shape is a trunk attention layer's, tensor
        // for tensor (compare `model.language_model.layers.3.self_attn.*` and
        // `layers.0.mlp.*`), which is what makes `Mtp` reuse `gated_attn` and
        // `dense_mlp` rather than restate them.
        //
        // **NO `mtp.lm_head` AND NO `mtp.embed_tokens`**, and the config says
        // so before the index does: `mtp_use_dedicated_embeddings: false`. The
        // draft readout goes through `lm_head.weight` and the draft embedding
        // through `model.language_model.embed_tokens.weight` — the base
        // planes, already claimed above, interned once by the recorder and
        // read by both heads.
        // **THE TOWER, PLANE FOR PLANE** (multimodal §2, campaign M-1/M-2).
        //
        // Every entry is a `copy`: the norms' scale and bias are applied by
        // ops rather than folded into the GEMMs behind them (§9.1 — half the
        // fold is expressible and the halves do not compose), and the position
        // table is gathered rather than baked (§11.3). So this contract states
        // no arithmetic at all, which is what an import contract should be.
        //
        // The ONE rewrite is the patch embed. `patch_embed.proj.weight` is
        // stored as a `Conv3d` kernel — `[hidden, C, T, P, P]` — and the plan
        // reads a matmul bank `[hidden, C·T·P²]` over the pre-unfolded patch
        // vectors the submission ships. Those are the same bytes in the same
        // order (the conv's own layout is channel-major then temporal then
        // spatial, which is the order `qwen_patchify_hwc` unfolds in), so it
        // is a `transmute` and not a transform.
        if let Some(t) = &self.tower {
            let v = |s: &str| format!("model.visual.{s}");
            tensors.push(declare(
                src,
                &t.patch_embed,
                flattened(src, v("patch_embed.proj.weight"), extents(&t.patch_embed))?,
            )?);
            tensors.push(copy(src, &t.patch_embed_bias, v("patch_embed.proj.bias"))?);
            tensors.push(copy(src, &t.pos_embed, v("pos_embed.weight"))?);
            for (l, b) in t.blocks.iter().enumerate() {
                let n = |s: &str| v(&format!("blocks.{l}.{s}"));
                for (weight, from) in [
                    (&b.norm1, n("norm1.weight")),
                    (&b.norm1_bias, n("norm1.bias")),
                    (&b.qkv, n("attn.qkv.weight")),
                    (&b.qkv_bias, n("attn.qkv.bias")),
                    (&b.proj, n("attn.proj.weight")),
                    (&b.proj_bias, n("attn.proj.bias")),
                    (&b.norm2, n("norm2.weight")),
                    (&b.norm2_bias, n("norm2.bias")),
                    (&b.fc1, n("mlp.linear_fc1.weight")),
                    (&b.fc1_bias, n("mlp.linear_fc1.bias")),
                    (&b.fc2, n("mlp.linear_fc2.weight")),
                    (&b.fc2_bias, n("mlp.linear_fc2.bias")),
                ] {
                    tensors.push(copy(src, weight, from)?);
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
                tensors.push(copy(src, weight, from)?);
            }
        }

        //
        // **AND AN OVERLAY HEAD IS THE SAME TWELVE MINUS THREE, UNDER ITS OWN
        // PREFIX** (campaign M-4, [`Recipe::Eagle`]). `pie model import <base>
        // --aux <head>` is family-BLIND — it copies a second checkpoint's
        // tensors into the same `.zt` with every name prefixed `aux.` — so
        // what makes those bytes a draft head is this block naming them, and
        // the spelling it names them by is the FAMILY'S OWN block spelling
        // (`layers.0.self_attn.q_proj.weight`, and so on) rather than a
        // fourth vocabulary. A head trained for this family is written in
        // that spelling already; a head that is not, is not this family's.
        //
        // [`Recipe::Eagle`]: super::model::Recipe::Eagle
        if let Some(mtp) = &self.mtp {
            let p = mtp.recipe.prefix();
            // The stored spelling under each recipe. MTP publishes one block
            // as `mtp.layers.0.*` inside the base checkpoint; an overlay
            // publishes the same block, and `--aux` gives it the prefix.
            let n = |s: &str| format!("{p}.layers.0.{s}");
            if let Some(pre) = &mtp.pre_fc {
                tensors.push(copy(
                    src,
                    &pre.embedding,
                    format!("{p}.pre_fc_norm_embedding.weight"),
                )?);
                tensors.push(copy(
                    src,
                    &pre.hidden,
                    format!("{p}.pre_fc_norm_hidden.weight"),
                )?);
            }

            // THE ONE STORED BANK, CUT AT ITS OWN SEAM. `mtp.fc.weight` is
            // `[hidden, 2·hidden]` and multiplies `[normed embedding | normed
            // hidden]`, so its columns `0..hidden` are the embedding half and
            // `hidden..2·hidden` the hidden half — the order dev concatenates
            // in (`launch_concat_bf16_rows(ws.q, ws.y, ...)`, `ws.q` holding
            // `rms(embed(tok))` and `ws.y` holding `rms(hidden)`). The slice
            // is the whole of the claim: no cast, no transpose, two contiguous
            // column bands of one tensor.
            let half = extents(&mtp.fc_embed)[1];
            let fc = format!("{p}.fc.weight");
            tensors.push(declare(
                src,
                &mtp.fc_embed,
                Expr::src(fc.clone()).slice(1, 0, half),
            )?);
            tensors.push(declare(
                src,
                &mtp.fc_hidden,
                Expr::src(fc).slice(1, half, half),
            )?);

            let a = &mtp.attn;
            tensors.push(copy(src, &mtp.mixer_norm, n("input_layernorm.weight"))?);
            tensors.push(copy(src, &a.qg_proj, n("self_attn.q_proj.weight"))?);
            tensors.push(copy(src, &a.k_proj, n("self_attn.k_proj.weight"))?);
            tensors.push(copy(src, &a.v_proj, n("self_attn.v_proj.weight"))?);
            tensors.push(copy(src, &a.o_proj, n("self_attn.o_proj.weight"))?);
            tensors.push(copy(src, &a.q_norm, n("self_attn.q_norm.weight"))?);
            tensors.push(copy(src, &a.k_norm, n("self_attn.k_norm.weight"))?);
            tensors.push(copy(
                src,
                &mtp.mlp_norm,
                n("post_attention_layernorm.weight"),
            )?);
            match &mtp.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?);
                    tensors.push(copy(src, down, n("mlp.down_proj.weight"))?);
                }
                Mlp::Routed { .. } => {
                    return Err(ModelError::Illegible {
                        name: n("mlp"),
                        detail: "a draft head is one block and routes to no experts".to_string(),
                    });
                }
            }
            if let Some(norm) = &mtp.norm {
                tensors.push(copy(src, norm, format!("{p}.norm.weight"))?);
            }
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        // **A NAMED REFUSAL, NOT A SILENT HALF-LOAD.** GGUF has no settled
        // spelling for this family's draft head — nothing in the cached
        // artifacts states one, and inventing `blk.*.nextn.*` here would
        // publish a contract whose names no converter writes and whose first
        // symptom is a load that lands fourteen planes and zeroes the
        // fifteenth. A SKU that declares a head and reads a file that cannot
        // state one is refused at the door, by name.
        // A tower is refused for the draft head's reason one door over: no GGUF
        // converter writes `model.visual.*` under any settled spelling, and
        // inventing one here would publish a contract whose first symptom is a
        // load that lands the trunk and zeroes a hundred and fifty planes.
        if self.tower.is_some() {
            return Err(ModelError::Illegible {
                name: "visual".to_string(),
                detail: "this SKU declares a vision tower and no GGUF spelling \
                         of one is settled; import it from the safetensors \
                         checkpoint"
                    .to_string(),
            });
        }
        if self.mtp.is_some() {
            return Err(ModelError::Illegible {
                name: "mtp".to_string(),
                detail: "this SKU declares an MTP draft head and no GGUF \
                         spelling of one is settled; import it from the \
                         safetensors checkpoint"
                    .to_string(),
            });
        }
        let mut tensors = vec![
            copy(src, &self.embed, "token_embd.weight")?,
            copy(src, &self.final_norm, "output_norm.weight")?,
        ];

        if let Head::Bank(head) = &self.head {
            tensors.push(copy(src, head, "output.weight")?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");

            tensors.push(copy(src, &w.mixer_norm, n("attn_norm.weight"))?);
            tensors.push(copy(src, &w.mlp_norm, n("ffn_norm.weight"))?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.push(copy(src, &a.qg_proj, n("attn_q.weight"))?);
                    tensors.push(copy(src, &a.k_proj, n("attn_k.weight"))?);
                    tensors.push(copy(src, &a.v_proj, n("attn_v.weight"))?);
                    tensors.push(copy(src, &a.o_proj, n("attn_output.weight"))?);
                    tensors.push(copy(src, &a.q_norm, n("attn_q_norm.weight"))?);
                    tensors.push(copy(src, &a.k_norm, n("attn_k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.push(copy(src, &g.in_qkvz, n("ssm_in.weight"))?);
                    tensors.push(copy(src, &g.in_ba, n("ssm_beta_alpha.weight"))?);
                    tensors.push(copy(src, &g.conv, n("ssm_conv1d.weight"))?);
                    tensors.push(copy(src, &g.dt_bias, n("ssm_dt.bias"))?);
                    tensors.push(copy(src, &g.a_log, n("ssm_a"))?);
                    tensors.push(copy(src, &g.norm, n("ssm_norm.weight"))?);
                    tensors.push(copy(src, &g.out_proj, n("ssm_out.weight"))?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("ffn_gate.weight"), n("ffn_up.weight")],
                    )?);
                    tensors.push(copy(src, down, n("ffn_down.weight"))?);
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
                    tensors.push(copy(src, router, n("ffn_gate_inp.weight"))?);

                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?);
                    tensors.push(copy(src, down, n("ffn_down_exps.weight"))?);
                    tensors.push(fused(
                        src,
                        shared_gate_up,
                        [n("ffn_gate_shexp.weight"), n("ffn_up_shexp.weight")],
                    )?);
                    tensors.push(copy(src, shared_down, n("ffn_down_shexp.weight"))?);
                    tensors.push(copy(src, shared_gate, n("ffn_gate_inp_shexp.weight"))?);
                }
            }
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }
}

/// The same tensor, re-typed to a stated shape — a `Conv3d` kernel read as the
/// matmul bank it already is.
///
/// **A TRANSMUTE AND NOT A TRANSFORM.** `[hidden, C, T, P, P]` in the file and
/// `[hidden, C·T·P²]` in the plan are the same bytes in the same order, so
/// this checks the element count and re-states the type. A mismatch is a
/// refusal here rather than a silently short read four stages later.
fn flattened(src: &ztensor::Source, from: String, want: Vec<i64>) -> Result<Expr, ModelError> {
    let Some(tensor) = src.get(&from) else {
        return Err(ModelError::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| ModelError::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let stored: i128 = shape.iter().map(|&n| i128::from(n)).product();
    let asked: i128 = want.iter().map(|&n| i128::from(n)).product();
    // **ASKED ONLY OF A SOURCE THAT STATES EXTENTS.** A name census — the
    // fixture `the_checkpoints_state_what_the_texts_read` writes out of an
    // index, which carries every tensor's NAME and rank and none of its
    // extents — states every shape as ones, and a count check against that
    // would refuse the very provenance it exists to verify. One element is not
    // a convolution kernel; it is a fixture saying it does not know, and the
    // honest answer is to let the plan compiler check it where the extents are
    // real (`checkpoint::plan::compile`, which is what a load and `identify`
    // both run). A source that DOES state extents is checked here, because
    // then the message can name both shapes.
    if stored > 1 && stored != asked {
        return Err(illegible(&format!(
            "is stored {shape:?} ({stored} elements) and the plan reads it as \
             {want:?} ({asked} elements)"
        )));
    }
    let part = tensor.part("data").map_err(|why| illegible(&why))?;
    let encoding = checkpoint::file::encoding_of(&tensor, &part).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(want, encoding)))
}

fn squeezed(src: &ztensor::Source, from: String) -> Result<Expr, ModelError> {
    let Some(tensor) = src.get(&from) else {
        return Err(ModelError::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| ModelError::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    // **BOTH SPELLINGS OF THE SAME BYTES.** Transformers stores a depthwise
    // conv1d `[channels, in/groups, kernel]` and MLX stores it
    // `[channels, kernel, in/groups]`; `in/groups` IS ONE for a depthwise
    // bank, so the two differ in which axis carries the 1 and in nothing
    // else — the `channels x kernel` values are contiguous in that order
    // either way, which is why this is a squeeze and not a transpose. Both
    // are accepted because both are published: the second is what
    // `mlx_lm.convert` writes.
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
