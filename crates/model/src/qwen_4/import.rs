//! The qwen4 import contracts: foreign checkpoints of `Qwen3.8-Flash-Next`,
//! read into the declaration `model.rs` states.
//!
//! Two layouts, `qwen_3::import`'s own pair: transformers
//! (`model.language_model.*`, `lm_head.weight`) and `mlx_lm`
//! (`language_model.model.*`, `language_model.lm_head.weight`). The MLX
//! spelling folds `+1.0` into every plain-RMSNorm plane — measured for this
//! family the way it was for qwen_3: the 4-bit conversion's `hc_norm` and
//! `q_norm` planes cluster around one where transformers' initialization
//! centres them at zero — and splits the fused expert banks at the seam
//! `read_concat` rejoins. What no layout of this family does is fuse the
//! GDN in-projections: both publish `in_proj_qkv | z | b | a` as four
//! tensors, so both spellings concatenate.
//!
//! **EVERY PROJECTION READS AS STORED** — the mixed-4/8 file's eight-bit
//! triplets land as the `MlxU8` planes `model.rs` declares, the four-bit
//! banks land as theirs, and the one weight transform this import states is
//! the norm fold above, taken back out. The dequantizing landing this file
//! shipped first (`Builder::read_dequant`, still the door for a scheme with
//! no kernel) was retired when the affine gemm point arrived: a text
//! declares what the file holds, and the device reads it there.
//!
//! **WHAT IS DELIBERATELY NOT READ**: `visual.*` (this SKU is text-only; a
//! vision row is a second declaration when a deployment wants one),
//! `mtp.*` (a draft arm this text does not yet carry), the
//! `self_attn.indexer.*` planes (the QSA cut — `model::Mixer::Attn`'s doc
//! names it), and the PLE's three hash buffers, which are derived and verified
//! rather than read (`model::Ple`'s doc says why).

use checkpoint::contract::{Expr, ModelContract};

use super::model::{Mixer, Mlp, Model};
use crate::qwen_3::import::squeezed;
use checkpoint_dsl::{Builder, Error};

#[derive(Clone, Copy)]
enum Layout {
    /// `model.language_model.*` + `lm_head.weight` — transformers.
    Transformers,
    /// `language_model.model.*` + `language_model.lm_head.weight` —
    /// `mlx_lm`. Whether the file is the mixed-4/8 stack or a plain bf16
    /// re-spelling is not asked here: each catalog row declares its own
    /// widths, a read that misses its triplet is a miss, and `identify`'s
    /// ladder hands the file to the row whose declaration it matches.
    Mlx,
}

impl Layout {
    fn embed(self) -> &'static str {
        match self {
            Self::Transformers => "model.language_model.embed_tokens.weight",
            Self::Mlx => "language_model.model.embed_tokens.weight",
        }
    }

    fn trunk(self, tail: &str) -> String {
        match self {
            Self::Transformers => format!("model.language_model.{tail}"),
            Self::Mlx => format!("language_model.model.{tail}"),
        }
    }

    fn lm_head(self) -> &'static str {
        match self {
            Self::Transformers => "lm_head.weight",
            Self::Mlx => "language_model.lm_head.weight",
        }
    }

    /// `mlx_lm`'s `sanitize` bakes the `+1.0` of every plain RMSNorm into
    /// the stored plane — `qwen_3::import`'s measurement, repeated on this
    /// family's own file.
    fn folds_the_norm_one(self) -> bool {
        match self {
            Self::Transformers => false,
            Self::Mlx => true,
        }
    }
}

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        for layout in [Layout::Transformers, Layout::Mlx] {
            if src.get(layout.embed()).is_some() {
                return self.import_from_safetensors(src, layout);
            }
        }
        Err(Error::Illegible {
            name: "qwen4".to_string(),
            detail: "neither safetensors spelling of the embedding is here; \
                     the two layouts this family reads are transformers and mlx_lm"
                .to_string(),
        })
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);

        // Undo the MLX fold on the planes the text scales by `weight + 1`.
        let norm = |from: String| -> Expr {
            let read = Expr::src(from);
            if layout.folds_the_norm_one() {
                read.bias(-1.0)
            } else {
                read
            }
        };

        b.read(&self.embed, layout.embed())?;
        b.read(&self.head, layout.lm_head())?;

        for (l, w) in self.layers.iter().enumerate() {
            let n = |tail: &str| layout.trunk(&format!("layers.{l}.{tail}"));

            match &w.mixer {
                Mixer::Attn(a) => {
                    // The stored `q_proj` is the fused query|gate bank —
                    // `[2 · q_heads · head_dim, hidden]`, `attn_output_gate`.
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
                    // The gated norm's weight is NOT plus-one-scaled, and MLX
                    // does not fold it — `qwen_3::import`'s ruling, held.
                    b.read(&g.norm, n("linear_attn.norm.weight"))?;
                    b.read(&g.out_proj, n("linear_attn.out_proj.weight"))?;
                }
            }

            let sites = [
                (&w.attn_res, "attn_hyper_connection"),
                (&w.mlp_res, "mlp_hyper_connection"),
            ];
            for (res, site) in sites {
                b.read_expr(&res.norm, norm(n(&format!("{site}.hc_norm.weight"))))?;
                b.read(&res.down, n(&format!("{site}.input_mix_weight_down.weight")))?;
                b.read(&res.up, n(&format!("{site}.input_mix_weight_up.weight")))?;
                if let Some(inject) = &res.inject {
                    b.read(inject, n(&format!("{site}.block_inject_weight.weight")))?;
                }
            }

            match &w.mlp {
                Mlp::Dense { .. } => unreachable!("every qwen4 layer routes"),
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
                    // The expert banks part the way qwen_3's do: transformers
                    // fuses gate|up, `mlx_lm` splits them into `switch_mlp`
                    // and `read_concat` rejoins the pair at the bank seam.
                    match layout {
                        Layout::Transformers => {
                            b.read(gate_up, n("mlp.experts.gate_up_proj"))?;
                            b.read(down, n("mlp.experts.down_proj"))?;
                        }
                        Layout::Mlx { .. } => {
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

        if let Some(p) = &self.ple {
            let n = |tail: &str| layout.trunk(&format!("layers.{}.ple.{tail}", p.layer));
            // The table: `split_ngram_parts` shards, rejoined at the seams
            // the declaration banded — one row space, the hashed offsets'
            // own.
            let shards: Vec<String> = (0..shard_count(p))
                .map(|i| n(&format!("ple_embedding.ngram_embedding.shard_{i}.weight")))
                .collect();
            b.read_concat(&p.table, shards)?;
            b.read(&p.key_proj, n("key_proj.weight"))?;
            b.read(&p.value_proj, n("value_proj.weight"))?;
            b.read_expr(&p.norm_key, norm(n("norm_key.weight")))?;
            b.read_expr(&p.norm_query, norm(n("norm_query.weight")))?;
            b.read_expr(&p.norm_conv, norm(n("norm_conv.weight")))?;
            b.read_expr(&p.conv, squeezed(src, n("conv1d.weight"))?)?;
        }

        b.read_expr(
            &self.mixer.norm,
            norm(layout.trunk("hyper_connection_mixer.hc_norm.weight")),
        )?;
        b.read(
            &self.mixer.down,
            layout.trunk("hyper_connection_mixer.input_mix_weight_down.weight"),
        )?;
        b.read(
            &self.mixer.up,
            layout.trunk("hyper_connection_mixer.input_mix_weight_up.weight"),
        )?;

        Ok(b.build())
    }
}

/// How many shards the table's band list declares — the weight's own seams,
/// counted rather than restated.
fn shard_count(p: &super::model::Ple) -> usize {
    match &p.table.shard {
        model_dsl::Shard::Cut { segments, .. } => segments.len(),
        model_dsl::Shard::Replicated => 1,
    }
}
