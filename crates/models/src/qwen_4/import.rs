//! Import contracts for qwen4 checkpoints (transformers and mlx_lm layouts) into the model.rs declaration.

use checkpoint::contract::{Expr, ModelContract};

use super::model::{Mixer, Mlp, Model};
use model_dsl::Weight;
use crate::qwen_3::import::squeezed;
use model_dsl::Platform;
use checkpoint_dsl::{Builder, Error};

#[derive(Clone, Copy)]
enum Layout {
    /// `model.language_model.*` + `lm_head.weight` — transformers.
    Transformers,
    /// `language_model.model.*` + `language_model.lm_head.weight` — mlx_lm.
    /// Layout doesn't distinguish dtype width; that's decided by which
    /// catalog row's declared widths the read matches.
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

    /// True for mlx_lm, which stores each plain-RMSNorm weight already offset by +1.0.
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
        for (what, layout) in [
            ("transformers", Layout::Transformers),
            ("mlx_lm", Layout::Mlx),
        ] {
            match self.import_from_safetensors(src, platform, layout) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {what}, {why}")),
            }
        }
        Err(Error::Illegible {
            name: "qwen4".to_string(),
            detail: format!(
                "no reading of this file lands every plane this family \
                 declares — {}",
                refusals.join("; "),
            ),
        })
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source, platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        for read in self.reads(layout) {
            match read {
                Read::One(w, name) => b.read(w, name)?,
                // read_concat also joins each part's .scales/.biases at the same seam.
                Read::Concat(w, names) => b.read_concat(w, names)?,
                // Undo the MLX fold on the planes the text scales by
                // `weight + 1`.
                Read::Norm(w, name) => {
                    let e = Expr::src(name);
                    let e = if layout.folds_the_norm_one() {
                        e.bias(-1.0)
                    } else {
                        e
                    };
                    b.read_expr(w, e)?;
                }
                Read::Squeeze(w, name) => b.read_expr(w, squeezed(src, name)?)?,
            }
        }
        Ok(b.build())
    }

    /// Every source name this import reads, in order, paired with the plane
    /// it lands in. Drives both the actual import and the census below, so
    /// they cannot drift; a joined bank appears once per stored part.
    #[must_use]
    pub fn mlx_planes(&self) -> Vec<(&Weight, String)> {
        self.reads(Layout::Mlx)
            .into_iter()
            .flat_map(|r| match r {
                Read::One(w, name) | Read::Norm(w, name) | Read::Squeeze(w, name) => {
                    vec![(w, name)]
                }
                Read::Concat(w, names) => names.into_iter().map(|n| (w, n)).collect(),
            })
            .collect()
    }

    /// The same census, names only — checked against a snapshot's `weight_map`.
    #[must_use]
    pub fn mlx_source_names(&self) -> Vec<String> {
        self.mlx_planes().into_iter().map(|(_, name)| name).collect()
    }

    // Deliberately not read: `visual.*`, `mtp.*`, `self_attn.indexer.*`, and
    // the PLE hash buffers (derived, not stored).
    fn reads(&self, layout: Layout) -> Vec<Read<'_>> {
        let mut reads = Vec::new();
        reads.push(Read::One(&self.embed, layout.embed().to_string()));
        reads.push(Read::One(&self.head, layout.lm_head().to_string()));

        for (l, w) in self.layers.iter().enumerate() {
            let n = |tail: &str| layout.trunk(&format!("layers.{l}.{tail}"));

            match &w.mixer {
                Mixer::Attn(a) => {
                    // Stored q_proj is the fused query|gate bank: [2 · q_heads · head_dim, hidden].
                    reads.push(Read::One(&a.qg_proj, n("self_attn.q_proj.weight")));
                    reads.push(Read::One(&a.k_proj, n("self_attn.k_proj.weight")));
                    reads.push(Read::One(&a.v_proj, n("self_attn.v_proj.weight")));
                    reads.push(Read::One(&a.o_proj, n("self_attn.o_proj.weight")));
                    reads.push(Read::Norm(&a.q_norm, n("self_attn.q_norm.weight")));
                    reads.push(Read::Norm(&a.k_norm, n("self_attn.k_norm.weight")));
                }
                Mixer::Gdn(g) => {
                    reads.push(Read::Concat(
                        &g.in_qkvz,
                        vec![
                            n("linear_attn.in_proj_qkv.weight"),
                            n("linear_attn.in_proj_z.weight"),
                        ],
                    ));
                    reads.push(Read::Concat(
                        &g.in_ba,
                        vec![
                            n("linear_attn.in_proj_b.weight"),
                            n("linear_attn.in_proj_a.weight"),
                        ],
                    ));
                    reads.push(Read::Squeeze(&g.conv, n("linear_attn.conv1d.weight")));
                    reads.push(Read::One(&g.dt_bias, n("linear_attn.dt_bias")));
                    reads.push(Read::One(&g.a_log, n("linear_attn.A_log")));
                    // Not plus-one-scaled; mlx_lm does not fold it either.
                    reads.push(Read::One(&g.norm, n("linear_attn.norm.weight")));
                    reads.push(Read::One(&g.out_proj, n("linear_attn.out_proj.weight")));
                }
            }

            let sites = [
                (&w.attn_res, "attn_hyper_connection"),
                (&w.mlp_res, "mlp_hyper_connection"),
            ];
            for (res, site) in sites {
                reads.push(Read::Norm(&res.norm, n(&format!("{site}.hc_norm.weight"))));
                reads.push(Read::One(
                    &res.down,
                    n(&format!("{site}.input_mix_weight_down.weight")),
                ));
                reads.push(Read::One(
                    &res.up,
                    n(&format!("{site}.input_mix_weight_up.weight")),
                ));
                if let Some(inject) = &res.inject {
                    reads.push(Read::One(
                        inject,
                        n(&format!("{site}.block_inject_weight.weight")),
                    ));
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
                    reads.push(Read::One(router, n("mlp.gate.weight")));
                    // transformers stores gate_up fused; mlx_lm splits it into
                    // switch_mlp.{gate,up}_proj and this rejoins them.
                    match layout {
                        Layout::Transformers => {
                            reads.push(Read::One(gate_up, n("mlp.experts.gate_up_proj")));
                            reads.push(Read::One(down, n("mlp.experts.down_proj")));
                        }
                        Layout::Mlx => {
                            reads.push(Read::Concat(
                                gate_up,
                                vec![
                                    n("mlp.switch_mlp.gate_proj.weight"),
                                    n("mlp.switch_mlp.up_proj.weight"),
                                ],
                            ));
                            reads.push(Read::One(down, n("mlp.switch_mlp.down_proj.weight")));
                        }
                    }
                    reads.push(Read::Concat(
                        shared_gate_up,
                        vec![
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    ));
                    reads.push(Read::One(
                        shared_down,
                        n("mlp.shared_expert.down_proj.weight"),
                    ));
                    reads.push(Read::One(shared_gate, n("mlp.shared_expert_gate.weight")));
                }
            }
        }

        if let Some(p) = &self.ple {
            let n = |tail: &str| layout.trunk(&format!("layers.{}.ple.{tail}", p.layer));
            // Stored sharded; shards are concatenated back into one row space.
            let shards: Vec<String> = (0..shard_count(p))
                .map(|i| n(&format!("ple_embedding.ngram_embedding.shard_{i}.weight")))
                .collect();
            reads.push(Read::Concat(&p.table, shards));
            reads.push(Read::One(&p.key_proj, n("key_proj.weight")));
            reads.push(Read::One(&p.value_proj, n("value_proj.weight")));
            reads.push(Read::Norm(&p.norm_key, n("norm_key.weight")));
            reads.push(Read::Norm(&p.norm_query, n("norm_query.weight")));
            reads.push(Read::Norm(&p.norm_conv, n("norm_conv.weight")));
            reads.push(Read::Squeeze(&p.conv, n("conv1d.weight")));
        }

        reads.push(Read::Norm(
            &self.mixer.norm,
            layout.trunk("hyper_connection_mixer.hc_norm.weight"),
        ));
        reads.push(Read::One(
            &self.mixer.down,
            layout.trunk("hyper_connection_mixer.input_mix_weight_down.weight"),
        ));
        reads.push(Read::One(
            &self.mixer.up,
            layout.trunk("hyper_connection_mixer.input_mix_weight_up.weight"),
        ));
        reads
    }
}

/// One read of this import: the plane it lands in, and the source name (or
/// names) it lands from.
enum Read<'a> {
    /// A plane read verbatim under its own name.
    One(&'a Weight, String),
    /// A bank joined from stored parts at the seams the declaration bands.
    Concat(&'a Weight, Vec<String>),
    /// A plain-RMSNorm plane, whose `+1` the MLX spelling folds in and this
    /// import takes back out.
    Norm(&'a Weight, String),
    /// A depthwise convolution stored with a unit axis the read squeezes.
    Squeeze(&'a Weight, String),
}

/// Number of shards declared by the table's own seams.
fn shard_count(p: &super::model::Ple) -> usize {
    match &p.table.shard {
        model_dsl::Shard::Cut { segments, .. } => segments.len(),
        model_dsl::Shard::Replicated => 1,
    }
}
