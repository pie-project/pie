use checkpoint::contract::{Expr, ModelContract};

use super::model::{AttnBanks, Model};
use checkpoint::contract::TensorType;

use checkpoint_dsl::{Builder, Error, extents};

/// **WHERE A SAFETENSORS CHECKPOINT OF THIS FAMILY PUTS ITS TRUNK.** Two
/// spellings, and the difference is one swapped pair of path components —
/// `qwen_3::import`'s `Layout` in gemma's spelling of it, for the same reason
/// and against the same `mlx_lm` habit.
///
/// Transformers publishes the multimodal wrapper first and the language tower
/// under it: `model.language_model.layers.*`. `mlx_lm`'s
/// `Gemma4Model.sanitize` strips the leading `model.` and then re-inserts it
/// one level down —
///
/// ```python
/// k = k.removeprefix("model.")
/// if k.startswith("language_model"):
///     k = k.replace("language_model.", "language_model.model.")
/// ```
///
/// — so the same tensors come back out as `language_model.model.layers.*`.
/// Nothing else moves: every leaf below `layers.{l}.` is spelled the same in
/// both, which is why this is a prefix and not a second import.
///
/// **THIS IS FIRST-LIGHT BREAKAGE, NOT A NEW FEATURE**, and the same breakage
/// `qwen_3` had: `gemma4-31b-mlxu4-kv-bf16` names the transformers spelling,
/// so no `mlx_lm` output had ever satisfied it — `import` refused at the door,
/// on the embedding, before one plane was read.
///
/// # What the audit did NOT find, and the finding is the point
///
/// **GEMMA 4 DOES NOT FOLD THE RMSNORM ONE, AND GEMMA 3 DID.** The rule
/// `qwen_3` left behind is that a family's `sanitize` is read before its first
/// load, and reading gemma's says the opposite of what the family's history
/// suggests. `mlx_lm/models/gemma3_text.py` carries its own `RMSNorm` class
/// computing `mx.fast.rms_norm(x, 1.0 + self.weight, eps)` — the gemma
/// convention through three generations. `gemma4_text.py` uses the stock
/// `nn.RMSNorm`, which is `x_norm * w` with no constant, and neither
/// `Gemma4Model.sanitize` nor `Gemma4TextModel.sanitize` touches a norm plane.
///
/// The checkpoint agrees, loudly: `layers.0.input_layernorm.weight` in
/// `mlx-community/gemma-4-31b-it-4bit` means +4.88 and reaches 444.0, and
/// `self_attn.q_norm.weight` is the constant 1.0234 across all 256 of its
/// entries. Those are multiplicative scales, not offsets from one. This
/// tree's `gemma_4::forward` already spells every one of them
/// `ops::elemwise::rmsnorm` — the plain kernel — so the reading is right and
/// there is NO `folds_the_norm_one` here. It is recorded because an absence
/// that was checked is worth as much as a presence that was found.
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

    /// **AND THE TWO MULTIMODAL NAMESPACES MOVE THE SAME WAY THE TRUNK DOES.**
    /// `Gemma4Model.sanitize`'s `k.removeprefix("model.")` runs over EVERY
    /// key and only the `language_model` branch is re-inserted, so
    /// transformers' `model.vision_tower.*` and `model.embed_vision.*` come
    /// out of `mlx_lm` as bare `vision_tower.*` and `embed_vision.*`. Read off
    /// `mlx-community/gemma-4-31b-it-4bit`'s index, where the trunk is
    /// `language_model.model.layers.*` and the tower is `vision_tower.*` in
    /// the same file.
    ///
    /// Nothing below the prefix moves, which is why this is one function and
    /// not a second import.
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
        let mut refusals: Vec<String> = Vec::new();
        for (what, layout) in [
            ("transformers", Layout::Transformers),
            ("mlx_lm", Layout::Mlx),
        ] {
            match self.import_from_safetensors(src, layout) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {what}, {why}")),
            }
        }
        match self.import_from_gguf(src) {
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
        src: &ztensor::Source,
    ) -> Result<ModelContract, Error> {
        self.import_from_safetensors(src, Layout::Transformers)
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
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
                    // **`attention_k_eq_v`: THE LAYER WITH NO VALUE
                    // PROJECTION, AND WHY READING THE KEY TWICE IS THE WHOLE
                    // OF IT.**
                    //
                    // `gemma-4-31b`'s ten global layers publish a `q_proj`, a
                    // `k_proj` and NO `v_proj` — 50 of the 60 layers hold one,
                    // and the ten that do not are exactly `layer_types`'
                    // `full_attention` entries. `config.text_config` says so
                    // before the index does: `attention_k_eq_v: true`.
                    // `mlx_lm/models/gemma4_text.py` builds the module the
                    // same way — `if not self.use_k_eq_v: self.v_proj = ...`
                    // — and its forward is one line:
                    //
                    // ```python
                    // keys = self.k_proj(x).reshape(...)
                    // values = keys
                    // ```
                    //
                    // The value stream IS the key stream, taken before either
                    // norm: `keys = self.k_norm(keys)` and
                    // `values = self.v_norm(values)` are applied to two
                    // references to one projection, and only the keys are
                    // rotated. `gemma_4::forward`'s `qkv_unfused` already
                    // computes exactly that shape — `rmsnorm_no_scale` on the
                    // v leg, `rmsnorm_per_head` with `k_norm` on the k leg,
                    // rope over `(q, k)` alone — so what the layer needs is
                    // not a new arm but the same bytes in both legs.
                    //
                    // So the v leg reads the KEY's triplet. Nothing about the
                    // declared bank changes: `qkv` is still
                    // `[q_w + 2·kv_w, hidden]` and `split_qkv` still cuts it
                    // where it always did. It costs the duplicate — 2048 rows
                    // of 5376 codes on ten layers of sixty, some 55 MiB — and
                    // buys a reading that is the checkpoint's own rather than
                    // an arm that has to be threaded through the model, the
                    // forward and the storage compiler to save it.
                    //
                    // **THE CHECKPOINT IS ASKED, NOT THE MODEL.** `k_eq_v` is
                    // a property of the artifact — the same `Model::b31`
                    // reads a transformers gemma-4-31b, which publishes all
                    // sixty `v_proj` — so the discriminator is whether this
                    // file holds one, which is what an import is for.
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

            // **THE ROUTED BRANCH** (`model::Moe`), where the checkpoint
            // ships one. Six names and one rewrite.
            //
            // **THE REWRITE IS THE ROUTER'S NORM.** `mlx_lm`'s `Router` reads
            // `mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)` —
            // the stored `router.scale` times `hidden**-0.5`, ONE gain handed
            // to one norm — so the constant belongs to the plane rather than
            // to the forward. It cannot be dropped: the norm's output feeds a
            // softmax through `router.proj`, so a uniform factor on it changes
            // which experts win by changing how sharp the distribution is.
            //
            // **AND THE EXPERT BANKS ARE SPLIT IN THIS LAYOUT**, the way
            // `qwen_3::import`'s are. `Gemma4TextModel.sanitize` cuts a
            // transformers `experts.gate_up_proj` into
            // `experts.switch_glu.gate_proj` and `.up_proj` on the way in, and
            // `mlx_lm.convert` writes the split form — `mlx-community/gemma-4
            // -26b-a4b-it-4bit` holds two `[128, 704, 352]` banks per layer
            // where a transformers file holds one `[128, 1408, 2816]`.
            // `read_concat` joins them on the axis `.bank([inter, inter])`
            // cut, gate first, carrying each part's `.scales` and `.biases` to
            // the same seams.
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

            // **THE PER-LAYER SCALAR, WHERE THE LAYER OWNS IT.** See
            // `model::Layer::scalar`: a stack with no per-layer embedding
            // still publishes one `layer_scalar` per layer and still
            // multiplies by it. A PLE stack's is claimed below instead,
            // because there it is the last term of the relay.
            if let Some(scalar) = &w.scalar {
                b.read(scalar, n("layer_scalar"))?;
            }
        }

        if let Some(ple) = &self.ple {
            b.read(&ple.model_proj, layout.at("per_layer_model_projection.weight"))?;
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

        // **THE TOWER, PLANE FOR PLANE** (multimodal §12, campaign M-2).
        //
        // Every entry a plain `read` but one: the norms are RMSNorm and applied by
        // ops, the clip bounds are read as `[1]` weights rather than baked
        // into a plan constant, and the only rewrite is the position table.
        // `position_embedding_table` is stored `[2, positions, hidden]` — an
        // x table and a y table — and the plan reads ONE `[2 · positions,
        // hidden]` bank so that a single `layout.embed_weighted` at two taps
        // answers `table[0][x] + table[1][y]`. Same bytes, same order: the
        // leading axis is contiguous, so it is a transmute and not a
        // transform.
        if let Some(t) = &self.tower {
            let v = |s: &str| layout.vision(s);
            b.read(&t.patch_embed, v("patch_embedder.input_proj.weight"))?;
            b.read_derived(&t.pos_embed, || {
                flattened(
                    src,
                    v("patch_embedder.position_embedding_table"),
                    extents(&t.pos_embed),
                )
            })?;
            b.read(&t.projection, layout.embed_vision())?;
            // **THE STANDARDIZATION**, when the tower states one. Two
            // `[hidden]` buffers, read whole — `Gemma4VisionModel.forward`'s
            // own `(h − std_bias) · std_scale`.
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
                    // **`.linear.` IS IN THE NAME EITHER WAY.**
                    // `Gemma4ClippableLinear` wraps an `nn.Linear` whatever
                    // `use_clipped_linears` says, so the bank sits one level
                    // down in both towers; what the flag decides is whether
                    // the four scalars stand beside it. Verified against
                    // `gemma-4-31b-it-4bit`, whose tower ships
                    // `...mlp.down_proj.linear.weight` and no `input_min`.
                    b.read(&c.bank, format!("{stem}.linear.weight"))?;
                    // The four bounds. Stored as rank-0 scalars, read as `[1]`.
                    if let Some(k) = &c.clip {
                        for (weight, suffix) in [
                            (&k.in_lo, "input_min"),
                            (&k.in_hi, "input_max"),
                            (&k.out_lo, "output_min"),
                            (&k.out_hi, "output_max"),
                        ] {
                            b.read_derived(weight, || {
                                flattened(src, format!("{stem}.{suffix}"), extents(weight))
                            })?;
                        }
                    }
                }
            }
        }

        // **THE OVERLAY HEAD** (campaign M-4). `pie model import --aux`
        // prefixes a second checkpoint's names with `aux.`, so what makes
        // those bytes a draft head is this block naming them — in the
        // FAMILY'S OWN block spelling, which is how a head trained for gemma
        // is written.
        if let Some(a) = &self.draft {
            // The one stored bank, cut at its own seam: `aux.fc.weight` is
            // `[hidden, 2·hidden]` over `[embedding | hidden]`, embedding
            // first.
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

        Ok(b.build())
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        // No GGUF converter writes `model.vision_tower.*` under any settled
        // spelling, and inventing one would publish a contract whose first
        // symptom is a load that lands the trunk and zeroes four hundred
        // planes.
        if self.draft.is_some() {
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
        // A named refusal for the reason the two above are named ones: the
        // routed branch is six planes a layer under a spelling no GGUF
        // converter of this family has settled, and a contract that skipped
        // them would load a stack whose second feedforward branch is zero —
        // fluent text, and not this model's.
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
        let mut b = Builder::new(src, self.tp);
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
            // See the safetensors door: a stack with no PLE owns its own
            // scalar, and a PLE stack's is claimed with the relay below.
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

/// The same tensor re-typed to a stated shape — a `[2, n, h]` pair of tables
/// read as one `[2n, h]` bank, and a rank-0 scalar read as `[1]`.
///
/// **A TRANSMUTE AND NOT A TRANSFORM**: the leading axis is contiguous, so the
/// bytes and their order are unchanged and this checks the element count and
/// restates the type. A source that states no extents — the name census
/// `the_checkpoints_state_what_the_texts_read` writes — is let through for
/// `qwen_3::import`'s reason, and the plan compiler checks it where the
/// extents are real.
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
    let part = tensor.part("data").map_err(|why| illegible(&why))?;
    let encoding = checkpoint::file::encoding_of(&tensor, &part).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(want, encoding)))
}
