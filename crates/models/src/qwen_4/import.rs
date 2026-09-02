//! Import contracts for qwen4 checkpoints (transformers and mlx_lm layouts) into the model.rs declaration.

use checkpoint::contract::{Expr, ModelContract};

use super::model::{Layer, Mixer, Mlp, Model};
use model_dsl::Weight;
use crate::qwen_3::import::{flattened, squeezed};
use model_dsl::Platform;
use checkpoint_dsl::{Builder, Error, extents};

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

    /// The draft head's planes sit at the top level in both spellings.
    fn mtp(self, tail: &str) -> String {
        format!("mtp.{tail}")
    }

    /// The tower is renamed, not just re-rooted: transformers publishes
    /// `model.visual.*`, `mlx_lm` publishes `vision_tower.*` (qwen_3's rule).
    fn tower(self, leaf: &str) -> String {
        match self {
            Self::Transformers => format!("model.visual.{leaf}"),
            Self::Mlx => format!("vision_tower.{leaf}"),
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
                // A raw plane stated as the same source stacked along its rows:
                // no shard seam to join, so a plain concat expression.
                Read::Stacked(w, names) => b.read_expr(
                    w,
                    Expr::concat(0, names.into_iter().map(Expr::src).collect()),
                )?,
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
                // `patch_embed.proj.weight` is a Conv3d kernel read as a matmul
                // bank `[hidden, C*T*P^2]`: a torch conv is already stored in
                // that byte order (a transmute); an MLX conv is channels-last,
                // so its columns need a permutation (qwen_3's reading).
                Read::PatchEmbed(w, name) => {
                    const CHANNELS: i64 = 3;
                    let want = extents(w);
                    let flat = flattened(src, name, want.clone())?;
                    let e = match layout {
                        Layout::Transformers => flat,
                        Layout::Mlx => {
                            let per = want[1] / CHANNELS;
                            let indices = (0..CHANNELS)
                                .flat_map(|c| (0..per).map(move |j| j * CHANNELS + c))
                                .collect();
                            flat.gather(1, indices)
                        }
                    };
                    b.read_expr(w, e)?;
                }
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
                Read::One(w, name)
                | Read::Norm(w, name)
                | Read::Squeeze(w, name)
                | Read::PatchEmbed(w, name) => vec![(w, name)],
                Read::Concat(w, names) | Read::Stacked(w, names) => {
                    names.into_iter().map(|n| (w, n)).collect()
                }
            })
            .collect()
    }

    /// The same census, names only — checked against a snapshot's `weight_map`.
    #[must_use]
    pub fn mlx_source_names(&self) -> Vec<String> {
        self.mlx_planes().into_iter().map(|(_, name)| name).collect()
    }

    // Deliberately not read: `self_attn.indexer.*` (the QSA indexer, which the
    // text does not run) and the PLE hash buffers (derived, not stored). The
    // tower and the draft head are read when the text declares them.
    fn reads(&self, layout: Layout) -> Vec<Read<'_>> {
        let mut reads = Vec::new();
        reads.push(Read::One(&self.embed, layout.embed().to_string()));
        reads.push(Read::One(&self.head, layout.lm_head().to_string()));

        for (l, w) in self.layers.iter().enumerate() {
            layer_reads(
                w,
                layout,
                &|tail: &str| layout.trunk(&format!("layers.{l}.{tail}")),
                &mut reads,
            );
        }

        if let Some(mtp) = &self.mtp {
            let n = |tail: &str| layout.mtp(tail);
            // Plain reads, NOT `Read::Norm`: the mlx conversion folded the
            // `+1` into every norm it ships EXCEPT these two (measured against
            // `Qwen/Qwen3.8-Flash-Next`: every `*norm.weight` differs from
            // the original by exactly 1.0, `pre_fc_norm_*` by 0.0 — the
            // converter's name pattern missed them). Both layouts therefore
            // hold the original's zero-centred weight, and the plus-one norm
            // in the forward puts the one back.
            reads.push(Read::One(&mtp.norm_embed, n("pre_fc_norm_embedding.weight")));
            reads.push(Read::One(&mtp.norm_hidden, n("pre_fc_norm_hidden.weight")));
            reads.push(Read::One(&mtp.fc_embed, n("fc_embedding.weight")));
            // `fc_hidden` applies per stream: the one stored plane, `streams`
            // times over, is the block-diagonal bank the text declares.
            reads.push(Read::Stacked(
                &mtp.fc_hidden,
                (0..self.streams).map(|_| n("fc_hidden.weight")).collect(),
            ));
            layer_reads(
                &mtp.block,
                layout,
                &|tail: &str| layout.mtp(&format!("layers.0.{tail}")),
                &mut reads,
            );
            reads.push(Read::Norm(&mtp.mixer.norm, n("hyper_connection_mixer.hc_norm.weight")));
            reads.push(Read::One(&mtp.mixer.down, n("hyper_connection_mixer.input_mix_weight_down.weight")));
            reads.push(Read::One(&mtp.mixer.up, n("hyper_connection_mixer.input_mix_weight_up.weight")));
        }

        if let Some(t) = &self.tower {
            let v = |leaf: &str| layout.tower(leaf);
            reads.push(Read::PatchEmbed(&t.patch_embed, v("patch_embed.proj.weight")));
            reads.push(Read::One(&t.patch_embed_bias, v("patch_embed.proj.bias")));
            reads.push(Read::One(&t.pos_embed, v("pos_embed.weight")));
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
                    reads.push(Read::One(weight, from));
                }
            }
            let mg = &t.merger;
            for (weight, from) in [
                (&mg.norm, v("merger.norm.weight")),
                (&mg.norm_bias, v("merger.norm.bias")),
                (&mg.fc1, v("merger.linear_fc1.weight")),
                (&mg.fc1_bias, v("merger.linear_fc1.bias")),
                (&mg.fc2, v("merger.linear_fc2.weight")),
                (&mg.fc2_bias, v("merger.linear_fc2.bias")),
            ] {
                reads.push(Read::One(weight, from));
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

/// One block's reads — the trunk's forty-eight and the draft head's one —
/// at the name prefix `n` states.
fn layer_reads<'a>(
    w: &'a Layer,
    layout: Layout,
    n: &dyn Fn(&str) -> String,
    reads: &mut Vec<Read<'a>>,
) {

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
    /// A raw plane read as one source repeated along axis 0 (`fc_hidden`'s
    /// per-stream bank).
    Stacked(&'a Weight, Vec<String>),
    /// The tower's Conv3d patch kernel, read as the matmul bank it is
    /// (columns permuted for the channels-last MLX spelling).
    PatchEmbed(&'a Weight, String),
}

/// Number of shards declared by the table's own seams.
fn shard_count(p: &super::model::Ple) -> usize {
    match &p.table.shard {
        model_dsl::Shard::Cut { segments, .. } => segments.len(),
        model_dsl::Shard::Replicated => 1,
    }
}
