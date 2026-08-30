use checkpoint::contract::{Expr, ModelContract};

use super::model::{AttnBanks, Model};
use checkpoint::contract::TensorType;

use crate::contract::{
    ALIGNMENT, ModelError, copy, declare, extents, fused, planes, planes_fused,
};

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
}

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        let gguf = "token_embd.weight";
        for layout in [Layout::Transformers, Layout::Mlx] {
            if src.get(&layout.embed()).is_some() {
                return self.import_from_safetensors(src, layout);
            }
        }
        if src.get(gguf).is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Illegible {
            name: "gemma4".to_string(),
            detail: format!(
                "it holds none of `{}`, `{}` or `{gguf}`",
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
        let mut tensors = planes(src, &self.embed, layout.embed())?;
        tensors.push(copy(src, &self.final_norm, layout.at("norm.weight"))?);

        for (l, w) in self.layers.iter().enumerate() {
            let n = |leaf: &str| layout.layer(l, leaf);

            tensors.push(copy(src, &w.attn_norm, n("input_layernorm.weight"))?);
            tensors.push(copy(
                src,
                &w.post_attn_norm,
                n("post_attention_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.pre_ffw_norm,
                n("pre_feedforward_layernorm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_ffw_norm,
                n("post_feedforward_layernorm.weight"),
            )?);
            tensors.push(copy(src, &w.attn.q_norm, n("self_attn.q_norm.weight"))?);
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    tensors.push(copy(src, k_norm, n("self_attn.k_norm.weight"))?);
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
                    tensors.extend(planes_fused(
                        src,
                        qkv,
                        [n("self_attn.q_proj.weight"), k, value],
                    )?);
                }

                AttnBanks::Shared { q_proj } => {
                    tensors.extend(planes(src, q_proj, n("self_attn.q_proj.weight"))?);
                }
            }
            tensors.extend(planes(src, &w.o_proj, n("self_attn.o_proj.weight"))?);
            tensors.extend(planes_fused(
                src,
                &w.gate_up,
                [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
            )?);
            tensors.extend(planes(src, &w.down, n("mlp.down_proj.weight"))?);

            // **THE PER-LAYER SCALAR, WHERE THE LAYER OWNS IT.** See
            // `model::Layer::scalar`: a stack with no per-layer embedding
            // still publishes one `layer_scalar` per layer and still
            // multiplies by it. A PLE stack's is claimed below instead,
            // because there it is the last term of the relay.
            if let Some(scalar) = &w.scalar {
                tensors.push(copy(src, scalar, n("layer_scalar"))?);
            }
        }

        if let Some(ple) = &self.ple {
            tensors.extend(planes(
                src,
                &ple.model_proj,
                layout.at("per_layer_model_projection.weight"),
            )?);
            tensors.push(copy(
                src,
                &ple.model_norm,
                layout.at("per_layer_projection_norm.weight"),
            )?);
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let n = |leaf: &str| layout.layer(l, leaf);
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                tensors.push(declare(
                    src,
                    &p.table,
                    Expr::src(layout.at("embed_tokens_per_layer.weight")).slice(1, at, width),
                )?);
                tensors.extend(planes(src, &p.gate, n("per_layer_input_gate.weight"))?);
                tensors.extend(planes(src, &p.proj, n("per_layer_projection.weight"))?);
                tensors.push(copy(src, &p.norm, n("post_per_layer_input_norm.weight"))?);

                tensors.push(copy(src, &p.scalar, n("layer_scalar"))?);
            }
        }

        // **THE TOWER, PLANE FOR PLANE** (multimodal §12, campaign M-2).
        //
        // Every entry a `copy` but one: the norms are RMSNorm and applied by
        // ops, the clip bounds are read as `[1]` weights rather than baked
        // into a plan constant, and the only rewrite is the position table.
        // `position_embedding_table` is stored `[2, positions, hidden]` — an
        // x table and a y table — and the plan reads ONE `[2 · positions,
        // hidden]` bank so that a single `layout.embed_weighted` at two taps
        // answers `table[0][x] + table[1][y]`. Same bytes, same order: the
        // leading axis is contiguous, so it is a transmute and not a
        // transform.
        if let Some(t) = &self.tower {
            let v = |s: &str| format!("model.vision_tower.{s}");
            tensors.push(copy(
                src,
                &t.patch_embed,
                v("patch_embedder.input_proj.weight"),
            )?);
            tensors.push(declare(
                src,
                &t.pos_embed,
                flattened(
                    src,
                    v("patch_embedder.position_embedding_table"),
                    extents(&t.pos_embed),
                )?,
            )?);
            tensors.push(copy(
                src,
                &t.projection,
                "model.embed_vision.embedding_projection.weight",
            )?);
            for (l, b) in t.blocks.iter().enumerate() {
                let n = |s: &str| v(&format!("encoder.layers.{l}.{s}"));
                for (weight, from) in [
                    (&b.attn_norm, n("input_layernorm.weight")),
                    (&b.post_attn_norm, n("post_attention_layernorm.weight")),
                    (&b.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
                    (&b.post_ffw_norm, n("post_feedforward_layernorm.weight")),
                    (&b.q_norm, n("self_attn.q_norm.weight")),
                    (&b.k_norm, n("self_attn.k_norm.weight")),
                ] {
                    tensors.push(copy(src, weight, from)?);
                }
                for (c, stem) in [
                    (&b.q, n("self_attn.q_proj")),
                    (&b.k, n("self_attn.k_proj")),
                    (&b.v, n("self_attn.v_proj")),
                    (&b.o, n("self_attn.o_proj")),
                    (&b.gate, n("mlp.gate_proj")),
                    (&b.up, n("mlp.up_proj")),
                    (&b.down, n("mlp.down_proj")),
                ] {
                    tensors.push(copy(src, &c.bank, format!("{stem}.linear.weight"))?);
                    // The four bounds. Stored as rank-0 scalars, read as `[1]`.
                    for (weight, suffix) in [
                        (&c.in_lo, "input_min"),
                        (&c.in_hi, "input_max"),
                        (&c.out_lo, "output_min"),
                        (&c.out_hi, "output_max"),
                    ] {
                        tensors.push(declare(
                            src,
                            weight,
                            flattened(src, format!("{stem}.{suffix}"), extents(weight))?,
                        )?);
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
            tensors.push(declare(
                src,
                &a.fc_embed,
                Expr::src("aux.fc.weight").slice(1, 0, half),
            )?);
            tensors.push(declare(
                src,
                &a.fc_hidden,
                Expr::src("aux.fc.weight").slice(1, half, half),
            )?);
            let n = |s: &str| format!("aux.layers.0.{s}");
            for (weight, from) in [
                (&a.attn_norm, n("input_layernorm.weight")),
                (&a.post_attn_norm, n("post_attention_layernorm.weight")),
                (&a.pre_ffw_norm, n("pre_feedforward_layernorm.weight")),
                (&a.post_ffw_norm, n("post_feedforward_layernorm.weight")),
                (&a.attn.q_norm, n("self_attn.q_norm.weight")),
                (&a.o_proj, n("self_attn.o_proj.weight")),
            ] {
                tensors.push(copy(src, weight, from)?);
            }
            if let AttnBanks::Owned { qkv, k_norm, .. } = &a.attn.banks {
                tensors.extend(planes_fused(
                    src,
                    qkv,
                    [
                        n("self_attn.q_proj.weight"),
                        n("self_attn.k_proj.weight"),
                        n("self_attn.v_proj.weight"),
                    ],
                )?);
                tensors.push(copy(src, k_norm, n("self_attn.k_norm.weight"))?);
            }
            tensors.push(fused(
                src,
                &a.gate_up,
                [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
            )?);
            tensors.push(copy(src, &a.down, n("mlp.down_proj.weight"))?);
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        // No GGUF converter writes `model.vision_tower.*` under any settled
        // spelling, and inventing one would publish a contract whose first
        // symptom is a load that lands the trunk and zeroes four hundred
        // planes.
        if self.draft.is_some() {
            return Err(ModelError::Illegible {
                name: "aux".to_string(),
                detail: "this SKU declares an aux draft head and no GGUF \
                         spelling of one is settled; import it from the \
                         safetensors artifact"
                    .to_string(),
            });
        }
        if self.tower.is_some() {
            return Err(ModelError::Illegible {
                name: "vision_tower".to_string(),
                detail: "this SKU declares a vision tower and no GGUF spelling \
                         of one is settled; import it from the safetensors \
                         checkpoint"
                    .to_string(),
            });
        }
        let mut tensors = Vec::new();

        tensors.push(copy(src, &self.embed, "token_embd.weight")?);
        tensors.push(copy(src, &self.final_norm, "output_norm.weight")?);

        for (l, w) in self.layers.iter().enumerate() {
            tensors.push(copy(
                src,
                &w.attn_norm,
                format!("blk.{l}.attn_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_attn_norm,
                format!("blk.{l}.post_attention_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.pre_ffw_norm,
                format!("blk.{l}.ffn_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.post_ffw_norm,
                format!("blk.{l}.post_ffw_norm.weight"),
            )?);
            tensors.push(copy(
                src,
                &w.attn.q_norm,
                format!("blk.{l}.attn_q_norm.weight"),
            )?);
            match &w.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    tensors.push(copy(src, k_norm, format!("blk.{l}.attn_k_norm.weight"))?);
                    tensors.push(fused(
                        src,
                        qkv,
                        [
                            format!("blk.{l}.attn_q.weight"),
                            format!("blk.{l}.attn_k.weight"),
                            format!("blk.{l}.attn_v.weight"),
                        ],
                    )?);
                }

                AttnBanks::Shared { q_proj } => {
                    tensors.push(copy(src, q_proj, format!("blk.{l}.attn_q.weight"))?);
                }
            }
            tensors.push(copy(src, &w.o_proj, format!("blk.{l}.attn_output.weight"))?);
            tensors.push(fused(
                src,
                &w.gate_up,
                [
                    format!("blk.{l}.ffn_gate.weight"),
                    format!("blk.{l}.ffn_up.weight"),
                ],
            )?);
            tensors.push(copy(src, &w.down, format!("blk.{l}.ffn_down.weight"))?);
            // See the safetensors door: a stack with no PLE owns its own
            // scalar, and a PLE stack's is claimed with the relay below.
            if let Some(scalar) = &w.scalar {
                tensors.push(copy(src, scalar, format!("blk.{l}.layer_scalar"))?);
            }
        }

        if let Some(ple) = &self.ple {
            tensors.push(copy(src, &ple.model_proj, "per_layer_model_proj.weight")?);
            tensors.push(copy(src, &ple.model_norm, "per_layer_proj_norm.weight")?);
            let width = i64::from(ple.dim);
            for (l, p) in ple.per_layer.iter().enumerate() {
                let at = i64::try_from(l).expect("a layer index inside i64") * width;
                tensors.push(declare(
                    src,
                    &p.table,
                    Expr::src("per_layer_token_embd.weight").slice(1, at, width),
                )?);
                tensors.push(copy(src, &p.gate, format!("blk.{l}.inp_gate.weight"))?);
                tensors.push(copy(src, &p.proj, format!("blk.{l}.proj.weight"))?);
                tensors.push(copy(src, &p.norm, format!("blk.{l}.post_norm.weight"))?);

                tensors.push(copy(src, &p.scalar, format!("blk.{l}.layer_scalar"))?);
            }
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
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
