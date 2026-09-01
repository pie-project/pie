use checkpoint::contract::{Expr, ModelContract, TensorType};

use super::model::{Head, Mixer, Mlp, Model};
use checkpoint_dsl::{Builder, Error, extents};

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

    /// **AND THE TOWER IS RENAMED, NOT JUST RE-ROOTED** — the one place this
    /// family's two spellings differ by more than a prefix swap. Transformers
    /// publishes `model.visual.*` and `mlx_lm` publishes `vision_tower.*`:
    /// the wrapper's own module name changes, not only its depth.
    ///
    /// Held against the two artifacts on this box —
    /// `Qwen/Qwen3.5-0.8B` and `mlx-community/Qwen3.5-0.8B-4bit` — the two
    /// namespaces hold **153 tensors each and the leaf sets are EQUAL**: same
    /// blocks, same merger, same `patch_embed`, same `pos_embed`, nothing in
    /// one and not the other. So this is a prefix and not a second import,
    /// exactly like the trunk's.
    ///
    /// **THIS WAS THE SAME FIRST-LIGHT BREAKAGE THE TRUNK HAD**, found the
    /// same way and one wave later: the tower block below named the
    /// transformers spelling alone, so no `mlx_lm` output could satisfy a
    /// `-vision-mlxu4` row — it refused at the door, on the patch embed,
    /// before one tower plane was read.
    fn tower(self, leaf: &str) -> String {
        match self {
            Self::Transformers => format!("model.visual.{leaf}"),
            Self::Mlx => format!("vision_tower.{leaf}"),
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
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        // **THE NATIVE DOOR, AND IT IS ASKED FIRST** (§M-4a). THE ARGUMENT IS
        // WRITTEN HERE AND THE OTHER SIX FAMILY TEXTS POINT AT IT.
        //
        // Every arm below sniffs a WITNESS NAME — the embedding, spelled the
        // way transformers or mlx_lm or GGUF spells it — and picks a layout
        // from it. That works because a foreign checkpoint's embedding is
        // under a foreign name. It stops working the moment `pie model import`
        // starts writing the embedding under `embed`: §M-4a promotes every
        // transform a contract states, and a source stored F16 or F32 states
        // one on EVERY plane it holds, so its artifact carries the whole model
        // under this text's own names and none of the three witnesses. The
        // sniff would then refuse an artifact this build had just written.
        //
        // [`Model::load`] IS the reader for that file — `read_own` throughout,
        // a straight read, no transform at all — and it was written before
        // anything called it. So it is asked first, and asking it is the whole
        // test: `load` succeeds exactly when the file holds EVERY plane this
        // contract declares under this contract's name, which is the fact that
        // makes a file a native artifact, and it fails on the first plane it
        // does not find. A foreign checkpoint pays one `claim` and one missing
        // lookup for the answer.
        //
        // **AND IT IS ALL OR NOTHING, WHICH IS WHY THE ARMS BELOW STILL RUN.**
        // An artifact is usually a MIXTURE: a promotion moves only the tensors
        // whose chain states a transform, so a bf16 checkpoint's artifact
        // holds the fused banks under `layer.N.qg_proj` and copies the rest
        // through under the source's spelling. `load` refuses that file and an
        // arm below reads it, with `Builder::holds_the_landed_plane` taking
        // the landed planes weight by weight as it goes. Two doors, and the
        // file decides which.
        if let Ok(native) = self.load(src) {
            return Ok(native);
        }
        // **AND THE ARM IS CHOSEN BY BUILDING IT, NOT BY SNIFFING A NAME**
        // — which is the same argument `load` above is, one level down, and
        // it is here because the sniff was measurably wrong.
        //
        // Each arm used to be gated on a WITNESS: the embedding, spelled the
        // way transformers or mlx_lm or GGUF spells it. The reasoning was
        // that a foreign checkpoint's embedding is under a foreign name, and
        // it holds right up until the embedding is one of the planes a
        // promotion MOVES. `mlx-community/Qwen3.5-0.8B-4bit` is that file: its
        // embedding is an affine triplet, whose chain states a `Transmute`,
        // so the artifact carries it as `embed` — while
        // `linear_attn.A_log` and `linear_attn.dt_bias` are stored at the
        // width this text declares, state no transform at all, and stay under
        // `language_model.model.layers.N.…`. 673 objects, 36 of them foreign,
        // and NEITHER DOOR OPENED IT: `load` failed on the 36, and the sniff
        // failed because the witness it was looking for had been promoted
        // away. `pie model import` wrote a file this same build then refused
        // to identify.
        //
        // A witness is a PROXY for "can this text read this file", and the
        // arm itself is the answer. It is cheap to ask — a contract build is
        // `src.get` and shape arithmetic, no payload — and it cannot go stale
        // the way a name can: an arm reads a mixture exactly when every plane
        // it needs is either landed under this text's name or present under
        // that layout's, which is the property the sniff was approximating.
        //
        // The order still decides ties, and there are none to decide. A file
        // that satisfies two arms at once would have to hold every plane of
        // one layout AND every plane of the other, or be fully landed — and a
        // fully landed file returned above.
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
                attempt(layout.spelling(), self.import_from_safetensors(src, layout))
            {
                return Ok(contract);
            }
        }
        if let Some(contract) = attempt("gguf", self.import_from_gguf(src)) {
            return Ok(contract);
        }
        Err(Error::Illegible {
            name: "qwen_3".to_string(),
            // Every arm's own refusal, which names the first plane it could
            // not find. The sniff could only ever say that three names were
            // absent; this says what each reading actually wanted.
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
    ) -> Result<ModelContract, Error> {
        self.import_from_safetensors(src, Layout::Transformers)
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        // One plain-RMSNorm plane, with the fold `mlx_lm` baked in taken
        // back out where the file carries one — see
        // [`Layout::folds_the_norm_one`]. Every read below that names a
        // plain-RMSNorm plane goes through this; the gated one does not, and
        // the projections never could.
        let norm = |from: String| -> Expr {
            let read = Expr::src(from);
            if layout.folds_the_norm_one() {
                read.bias(-1.0)
            } else {
                read
            }
        };

        let mut b = Builder::new(src, self.tp);
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
                    projection(&mut b, &a.qg_proj, n("self_attn.q_proj.weight"))?;
                    projection(&mut b, &a.k_proj, n("self_attn.k_proj.weight"))?;
                    projection(&mut b, &a.v_proj, n("self_attn.v_proj.weight"))?;
                    projection(&mut b, &a.o_proj, n("self_attn.o_proj.weight"))?;
                    b.read_expr(&a.q_norm, norm(n("self_attn.q_norm.weight")))?;
                    b.read_expr(&a.k_norm, norm(n("self_attn.k_norm.weight")))?;
                }
                Mixer::Gdn(g) => {
                    projection_concat(
                        &mut b,
                        &g.in_qkvz,
                        [n("linear_attn.in_proj_qkv.weight"), n("linear_attn.in_proj_z.weight")],
                    )?;

                    projection_concat(
                        &mut b,
                        &g.in_ba,
                        [n("linear_attn.in_proj_b.weight"), n("linear_attn.in_proj_a.weight")],
                    )?;

                    b.read_derived(&g.conv, || {
                        squeezed(src, n("linear_attn.conv1d.weight"))
                    })?;

                    b.read(&g.dt_bias, n("linear_attn.dt_bias"))?;
                    b.read(&g.a_log, n("linear_attn.A_log"))?;
                    b.read(&g.norm, n("linear_attn.norm.weight"))?;
                    projection(&mut b, &g.out_proj, n("linear_attn.out_proj.weight"))?;
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    projection_concat(
                        &mut b,
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?;
                    projection(&mut b, down, n("mlp.down_proj.weight"))?;
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
                    // The routing gates are named the same in both spellings;
                    // what differs is how many TENSORS each is and how wide
                    // its codes are, and both of those are facts about the
                    // weight's declared dtype rather than about the layout.
                    // `Model::new` declares them `U8g64` wherever the stack is
                    // `U4g64`, and `read` asks the weight rather than the file.
                    b.read(router, n("mlp.gate.weight"))?;

                    // **THE ONE PLACE THE TWO SPELLINGS PART** — `gpt_oss::
                    // import`'s `Layout` match, in this family's names.
                    //
                    // transformers ships the routed gate and up FUSED, one
                    // `mlp.experts.gate_up_proj` of `[experts, 2 * inter,
                    // hidden]`. `mlx_lm`'s `Qwen3_5MoeModel.sanitize` splits
                    // that pair — `gate_up[..., :mid, :]` and `gate_up[...,
                    // mid:, :]` — into `switch_mlp.gate_proj` and
                    // `switch_mlp.up_proj`, and `mlx_lm.convert` runs
                    // `sanitize` on the way in and writes the SPLIT form, so
                    // the split is a permanent property of every MLX-layout
                    // artifact of this family. `mlx-community/Qwen3.6-35B-A3B
                    // -4bit` holds two `[256, 512, 2048]` banks per layer where
                    // a transformers file holds one `[256, 1024, 2048]`.
                    //
                    // `read_concat` joins them on the weight's own cut axis,
                    // which `.bank([inter, inter])` put at axis 1 — the axis
                    // `sanitize` split and in the gate-first order it split
                    // them into — and carries each part's `.scales` and
                    // `.biases` to the same seams, a group belonging to the row
                    // it scales.
                    //
                    // `down_proj` is one tensor in both spellings and parts
                    // only in its name.
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
                    projection_concat(
                        &mut b,
                        shared_gate_up,
                        [
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    )?;
                    projection(&mut b, shared_down, n("mlp.shared_expert.down_proj.weight"))?;
                    b.read(shared_gate, n("mlp.shared_expert_gate.weight"))?;
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
        // Every entry is a plain `read`: the norms' scale and bias are applied by
        // ops rather than folded into the GEMMs behind them (§9.1 — half the
        // fold is expressible and the halves do not compose), and the position
        // table is gathered rather than baked (§11.3). So this contract states
        // no arithmetic at all, which is what an import contract should be.
        //
        // The ONE rewrite is the patch embed. `patch_embed.proj.weight` is a
        // `Conv3d` kernel, and the plan reads a matmul bank `[hidden, C·T·P²]`
        // over the pre-unfolded patch vectors the submission ships. A torch
        // conv stores channel-major `[hidden, C, T, P, P]` — the same bytes in
        // the same order as `qwen_patchify_hwc` unfolds, a transmute. An MLX
        // conv stores channels-last `[hidden, T, P, P, C]` — the same element
        // COUNT, which is all a transmute checks, and the picture read down
        // the wrong axis: three solid colours all captioned " black". Lane k
        // of the stored MLX row is ((t·P + r)·P + q)·C + c, so channel c's
        // T·P² lanes are the stride-C progression starting at c, already in
        // (t, r, q) order — three of those, joined, is the channel-major bank.
        if let Some(t) = &self.tower {
            let v = |s: &str| layout.tower(s);
            const CHANNELS: i64 = 3;
            let want = extents(&t.patch_embed);
            b.read_derived(&t.patch_embed, || {
                let flat = flattened(src, v("patch_embed.proj.weight"), want.clone())?;
                Ok(match layout {
                    Layout::Transformers => flat,
                    Layout::Mlx => {
                        // One permutation, not three progressions: a stride
                        // per channel breaks into an element-sized copy list
                        // the contract refuses at a million stretches; the
                        // gather is the lowering that refusal names.
                        let per = want[1] / CHANNELS;
                        let indices = (0..CHANNELS)
                            .flat_map(|c| (0..per).map(move |j| j * CHANNELS + c))
                            .collect();
                        flat.gather(1, indices)
                    }
                })
            })?;
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
                b.read(&pre.embedding, format!("{p}.pre_fc_norm_embedding.weight"))?;
                b.read(&pre.hidden, format!("{p}.pre_fc_norm_hidden.weight"))?;
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

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
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
        let mut b = Builder::new(src, self.tp);
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
/// matmul bank it already is.
///
/// **A TRANSMUTE AND NOT A TRANSFORM.** `[hidden, C, T, P, P]` in the file and
/// `[hidden, C·T·P²]` in the plan are the same bytes in the same order, so
/// this checks the element count and re-states the type. A mismatch is a
/// refusal here rather than a silently short read four stages later.
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

/// **A DENSE PROJECTION, READ OUT OF WHICHEVER ORDER THE FILE HOLDS** (§J4b).
///
/// Two arms, and the weight's own declaration picks between them: `U4g64tiled`
/// states the relabelling for `pie model import` to run (and is refused, by
/// name, on any serving load that reaches it), and every other width reads
/// straight through.
///
/// **THERE WAS A THIRD ARM AND IT MOVED** (§M-4a). It opened
/// `if src.get(&w.name).is_some() { b.read_own(w) }` — a file holding this
/// tensor under the WEIGHT'S name is an artifact `pie model import` wrote out
/// of this very contract, so the repack has run and the plane binds with no
/// transform. That reading did not change; what changed is that §M-4a
/// promotes the whole landing and not just the repack, so the arm has to hold
/// for every weight this text reads and not only for the ones routed through
/// here. It is now [`Builder::holds_the_landed_plane`], asked by every verb,
/// and this function is two arms because the first one is answered before it
/// is reached.
fn projection(
    b: &mut Builder,
    w: &model_dsl::Weight,
    from: String,
) -> Result<(), Error> {
    if w.dtype == model_dsl::Dtype::U4g64tiled {
        return b.read_repack(w, from);
    }
    b.read(w, from)
}

/// [`projection`] over legs the checkpoint ships apart — the join first and
/// the relabelling second, which is the only order that means anything
/// ([`Builder::read_repack_concat`]).
fn projection_concat(
    b: &mut Builder,
    w: &model_dsl::Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<(), Error> {
    if w.dtype == model_dsl::Dtype::U4g64tiled {
        return b.read_repack_concat(w, parts);
    }
    b.read_concat(w, parts)
}

fn extent(of: u64) -> i64 {
    i64::try_from(of).expect("an extent no i64 holds")
}
